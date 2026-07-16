use std::{cell::RefCell, collections::BTreeMap, ffi::c_void, rc::Rc, sync::Arc};

use bullet_compiler::{
    ir::NodeId,
    rewriterule,
    tensor::{
        DType, IRTrace, OpType, Size, TType, TensorIR, TensorOp,
        operation::{
            BroadcastAcrossDimension, CABinary, CABinaryOp, Matmul, MatrixLayout, PadAcrossDimension,
            ReduceAcrossDimension, Reduction, ScalarConstant, SliceAcrossDimension,
        },
        transform::{
            IRTransform,
            eliminate::{EliminateCommonSubExpressions, EliminateUnusedOperations},
            modify::AddOperation,
            rewriterules::RewritePass,
        },
    },
};

use crate::{
    buffer::{Buffer, SyncOnDrop, SyncOnValue},
    kernel::KernelSrc,
    pointwise::transforms::{CodegenPointwise, FusePointwise, LowerPointwise},
    runtime::{Blas, Device, Dim3, GemmConfig, Gpu, Kernel, Module, Stream},
};

#[derive(Clone, Copy, Debug)]
pub struct FunctionInput {
    idx: usize,
    is_mut: bool,
    ty: TType,
}

#[derive(Debug)]
enum Arg {
    Pointer { idx: usize },
    Size(Size),
}

enum Inst<G: Gpu> {
    Malloc {
        idx: usize,
        ty: TType,
    },
    Free {
        _idx: usize,
    },
    Zero {
        idx: usize,
        ty: TType,
    },
    LaunchKernel {
        func: Kernel<G>,
        args: Vec<Arg>,
        gdim: Rc<dyn Fn(usize) -> Dim3>,
        bdim: Rc<dyn Fn(usize) -> u32>,
        smem: Rc<dyn Fn(usize) -> u32>,
    },
    Matmul {
        cfg: Matmul,
        a: usize,
        b: usize,
        c: usize,
    },
}

pub struct Function<G: Gpu> {
    device: Arc<Device<G>>,
    maps: Box<[(NodeId, FunctionInput)]>,
    insts: Box<[Inst<G>]>,
    num_ptrs: usize,
    blas: Option<Blas<G>>,
    prealloc_size: usize,
    preallocs: Vec<Option<G::DevicePtr>>,
    scratch: RefCell<FunctionScratch<G>>,
}

struct FunctionScratch<G: Gpu> {
    ptrs: Vec<G::DevicePtr>,
    aliases: Vec<(G::DevicePtr, bool)>,
    sizes: Vec<i32>,
    kernel_args: Vec<*mut c_void>,
}

impl<G: Gpu> FunctionScratch<G> {
    fn new(num_ptrs: usize, max_num_args: usize) -> Self {
        Self {
            ptrs: vec![G::DevicePtr::default(); num_ptrs],
            aliases: Vec::new(),
            sizes: Vec::with_capacity(max_num_args),
            kernel_args: Vec::with_capacity(max_num_args),
        }
    }

    fn prepare(&mut self, num_ptrs: usize) {
        debug_assert_eq!(self.ptrs.len(), num_ptrs);
        self.aliases.clear();
        self.sizes.clear();
        self.kernel_args.clear();
    }
}

impl<G: Gpu> Drop for Function<G> {
    fn drop(&mut self) {
        let _ = self.dealloc_preallocs();
    }
}

impl<G: Gpu> Function<G> {
    pub fn dealloc_preallocs(&mut self) -> Result<(), G::Error> {
        if self.prealloc_size == 0 {
            return Ok(());
        }

        for &ptr in self.preallocs.iter().flatten() {
            unsafe { self.device.free(ptr)? };
        }

        self.prealloc_size = 0;
        self.preallocs.clear();

        Ok(())
    }

    pub fn new(device: Arc<Device<G>>, mut ir: TensorIR) -> Result<Self, IRTrace> {
        let props = device.props().clone();
        ir.transform(RewritePass(MatmulToBroadcastMul))?;
        ir.transform(DuplicateScalarsAndIndexing)?;
        ir.transform(LowerPointwise(props.clone()))?;
        ir.transform(FusePointwise(props.clone()))?;
        ir.transform(RewritePass(ReduceToMatmul))?;
        ir.transform(EliminateCommonSubExpressions)?;
        ir.transform(LowerPointwise(props))?;
        ir.transform(CodegenPointwise)?;
        ir.transform(CodegenReduction)?;

        let mut maps = BTreeMap::new();
        let mut num_ptrs = 0;
        let mut insts = Vec::new();
        let mut requires_blas = false;

        let mut times_seen = BTreeMap::new();
        let mut indices = BTreeMap::new();

        let mut max_num_args = 0;

        for op in ir.ordered_operations()? {
            // allocate output buffers
            for &output in op.outputs() {
                if ir.is_input(output)? {
                    let input = op.outputs()[0];
                    maps.insert(input, FunctionInput { idx: num_ptrs, is_mut: false, ty: ir.get_node(input)?.ty() });
                } else if ir.is_output(output) {
                    maps.insert(output, FunctionInput { idx: num_ptrs, is_mut: true, ty: ir.get_node(output)?.ty() });
                } else {
                    insts.push(Inst::Malloc { idx: num_ptrs, ty: ir.get_node(output)?.ty() });
                    times_seen.insert(output, 0);
                }

                indices.insert(output, num_ptrs);
                num_ptrs += 1;
            }

            // insert kernels
            let data = op.data();
            if let Some(KernelSrc {
                name,
                source,
                requires_var_size_arg,
                arg_order,
                gdim,
                bdim,
                smem,
                requires_zero,
                ..
            }) = data.downcast().cloned()
            {
                let mut args = Vec::new();

                if requires_var_size_arg {
                    args.push(Arg::Size(Size::variable()));
                }

                for (index, is_input) in arg_order {
                    let node_id = if is_input { op.inputs()[index] } else { op.outputs()[index] };
                    args.push(Arg::Pointer { idx: *indices.get(&node_id).unwrap() });
                }

                for output in requires_zero {
                    let node_id = op.outputs()[output];
                    let ty = ir.get_node(node_id)?.ty();
                    insts.push(Inst::Zero { idx: *indices.get(&node_id).unwrap(), ty });
                }

                let func = Module::new(device.clone(), source.clone())
                    .map_err(|e| IRTrace::from(format!("{e:?}\n{source}")))?
                    .get_kernel(name)
                    .map_err(|e| IRTrace::from(format!("{e:?}\n{source}")))?;

                max_num_args = max_num_args.max(args.len());
                insts.push(Inst::LaunchKernel { func, args, gdim, bdim, smem });
            } else if let Some(cfg) = data.downcast::<Matmul>().cloned() {
                if cfg.dtype != DType::F32 {
                    return Err("Unsupported matmul dtype!".into());
                }

                let [a, b] = op.inputs()[..] else { return Err("Invalid inputs!".into()) };
                let [c] = op.outputs()[..] else { return Err("Invalid inputs!".into()) };

                requires_blas = true;
                insts.push(Inst::Matmul {
                    cfg,
                    a: *indices.get(&a).unwrap(),
                    b: *indices.get(&b).unwrap(),
                    c: *indices.get(&c).unwrap(),
                })
            } else if !data.is_input() {
                return Err(format!("Unsupported operation: {data:?}").into());
            }

            // free buffers that see no more usage
            for &input in op.inputs() {
                if !ir.is_input(input)? && !ir.is_output(input) {
                    let times_seen = times_seen.get_mut(&input).unwrap();
                    *times_seen += 1;

                    if ir.get_node(input)?.children() == *times_seen {
                        let idx = *indices.get(&input).unwrap();
                        insts.push(Inst::Free { _idx: idx });
                    }
                }
            }
        }

        let blas = requires_blas.then(|| Blas::new(device.clone()).unwrap());
        Ok(Self {
            device,
            maps: maps.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            insts: insts.into_boxed_slice(),
            num_ptrs,
            blas,
            prealloc_size: 0,
            preallocs: Vec::new(),
            scratch: RefCell::new(FunctionScratch::new(num_ptrs, max_num_args)),
        })
    }

    pub fn prealloc(&mut self, var_size: usize) -> Result<(), G::Error> {
        if var_size == self.prealloc_size {
            return Ok(());
        }

        self.dealloc_preallocs()?;
        self.preallocs = vec![None; self.num_ptrs];

        for inst in &self.insts {
            if let &Inst::Malloc { idx, ty } = inst {
                let bytes = ty.dtype().bytes() * ty.size().evaluate(var_size);
                self.preallocs[idx] = Some(self.device.malloc(bytes)?);
            }
        }

        self.prealloc_size = var_size;

        Ok(())
    }

    pub fn execute(
        &self,
        stream: Arc<Stream<G>>,
        inputs: &BTreeMap<NodeId, Arc<Buffer<G>>>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
        self.execute_binding_refs(stream, inputs.iter().map(|(&node, buffer)| (node, buffer)))
    }

    pub fn execute_bindings(
        &self,
        stream: Arc<Stream<G>>,
        inputs: impl IntoIterator<Item = (NodeId, Arc<Buffer<G>>)>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        self.execute_binding_refs(stream, inputs.iter().map(|(node, buffer)| (*node, buffer)))
    }

    pub fn execute_binding_refs<'a>(
        &self,
        stream: Arc<Stream<G>>,
        inputs: impl IntoIterator<Item = (NodeId, &'a Arc<Buffer<G>>)>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error>
    where
        G: 'a,
    {
        let inputs = inputs
            .into_iter()
            .map(|(name, buffer)| -> Result<_, G::Error> {
                let input = match self.input_slot(name) {
                    Some(input) => input,
                    None => return Err(String::from("Input not in function!").into()),
                };
                Ok((input, buffer))
            })
            .collect::<Result<Vec<_>, G::Error>>()?;
        self.execute_resolved_binding_refs(stream, inputs)
    }

    pub fn input_slot(&self, node: NodeId) -> Option<FunctionInput> {
        self.maps
            .binary_search_by_key(&node, |(node, _)| *node)
            .ok()
            .map(|idx| self.maps[idx].1)
    }

    pub fn execute_resolved_binding_refs<'a>(
        &self,
        stream: Arc<Stream<G>>,
        inputs: impl IntoIterator<Item = (FunctionInput, &'a Arc<Buffer<G>>)>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error>
    where
        G: 'a,
    {
        let inputs = inputs.into_iter();
        let input_capacity = inputs.size_hint().0;
        let mut sync = SyncOnDrop::with_capacity(stream.clone(), input_capacity);
        let mut scratch = self.scratch.borrow_mut();
        scratch.prepare(self.num_ptrs);

        let aliases_capacity = scratch.aliases.capacity();
        if aliases_capacity < input_capacity {
            scratch.aliases.reserve(input_capacity - aliases_capacity);
        }
        let mut var_size = None;

        for (input, buf) in inputs {
            let FunctionInput { idx, is_mut, ty } = input;
            let size = ty.size();

            if buf.dtype() != ty.dtype() {
                return Err("Mismatched dtypes!".to_string().into());
            }

            if let Some(new_var) = size.get_var_size(buf.size()) {
                if size.evaluate(new_var) != buf.size() {
                    return Err("Mismatched sizes!".to_string().into());
                }

                match var_size {
                    None => var_size = Some(new_var),
                    Some(old_var) => {
                        if old_var != new_var {
                            return Err("Mismatched var sizes!".to_string().into());
                        }
                    }
                }
            } else {
                match size.evaluate_constant() {
                    None => return Err("Invalid var size!".to_string().into()),
                    Some(len) => {
                        if len != buf.size() {
                            return Err("Mismatched sizes!".to_string().into());
                        }
                    }
                }
            }

            let guard = buf.acquire(stream.clone())?;
            let ptr = guard.ptr();
            sync.attach(guard)?;
            scratch.ptrs[idx] = ptr;

            if let Some((_, is_alr_mut)) = scratch.aliases.iter().find(|(seen_ptr, _)| *seen_ptr == ptr) {
                if is_mut || *is_alr_mut {
                    return Err("Cannot alias pointers!".to_string().into());
                }
            } else {
                scratch.aliases.push((ptr, is_mut));
            }
        }

        let var = var_size.unwrap_or(1);

        assert_ne!(var, 0, "Variable size = 0!");
        assert!(self.prealloc_size >= var);

        unsafe {
            for inst in &self.insts {
                match inst {
                    &Inst::Malloc { idx, .. } => {
                        scratch.ptrs[idx] = self.preallocs[idx].expect("Missing preallocated buffer");
                    }
                    &Inst::Zero { idx, ty } => {
                        let bytes = ty.size().evaluate(var) * ty.dtype().bytes();
                        stream.memset(scratch.ptrs[idx], bytes, 0)?;
                    }
                    Inst::Free { .. } => {}
                    Inst::LaunchKernel { func, args, gdim, bdim, smem } => {
                        scratch.kernel_args.clear();
                        scratch.sizes.clear();
                        for arg in args {
                            let ptrs = scratch.ptrs.as_ptr();
                            let arg_ptr = match arg {
                                Arg::Pointer { idx } => ptrs.add(*idx).cast_mut().cast(),
                                Arg::Size(size) => {
                                    scratch.sizes.push(size.evaluate(var) as i32);
                                    let last = scratch.sizes.last().unwrap();
                                    (last as *const i32).cast_mut().cast()
                                }
                            };
                            scratch.kernel_args.push(arg_ptr);
                        }

                        func.launch(&stream, gdim(var), bdim(var), scratch.kernel_args.as_mut_ptr(), smem(var))?;
                    }
                    &Inst::Matmul { cfg, a, b, c } => {
                        let handle = self.blas.as_ref().unwrap();
                        let config = GemmConfig {
                            row_mjr_a: !cfg.lhs.col_mjr,
                            row_mjr_b: !cfg.rhs.col_mjr,
                            m: cfg.lhs.rows.evaluate(var).try_into().unwrap(),
                            n: cfg.rhs.cols.evaluate(var).try_into().unwrap(),
                            k: cfg.lhs.cols.evaluate(var).try_into().unwrap(),
                            alpha: 1.0,
                            beta: 0.0,
                        };

                        if let Some(1) = cfg.batch.evaluate_constant() {
                            handle.gemm(stream.as_ref(), config, scratch.ptrs[a], scratch.ptrs[b], scratch.ptrs[c])?;
                        } else {
                            let batch = cfg.batch.evaluate(var);
                            handle.batched_gemm(
                                stream.as_ref(),
                                batch,
                                config,
                                scratch.ptrs[a],
                                scratch.ptrs[b],
                                scratch.ptrs[c],
                            )?;
                        }
                    }
                }
            }
        }

        Ok(SyncOnValue::new(sync, self))
    }
}

/// Separate out all `ScalarConst`s, as otherwise we end up
/// materialising them in kernel A and passing to kernel B,
/// rather than handling internally for each
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DuplicateScalarsAndIndexing;
impl IRTransform for DuplicateScalarsAndIndexing {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            for &input in op.inputs() {
                let grandparents = ir.get_op(ir.get_parent_op(input)?)?.inputs().to_vec();

                if let Some(&ScalarConstant(value, size)) = ir.parent_op(input)? {
                    let new_scalar = ir.add_scalar(value, size);
                    ir.ir_mut().replace_single_input(op.id(), new_scalar, input)?;
                } else if let Some(broadcast) = ir.parent_op::<BroadcastAcrossDimension>(input)? {
                    let broadcast = ir.add_op(grandparents, Ok::<_, IRTrace>(*broadcast))?[0];
                    ir.ir_mut().replace_single_input(op.id(), broadcast, input)?;
                } else if let Some(slice) = ir.parent_op::<SliceAcrossDimension>(input)? {
                    let slice = ir.add_op(grandparents, Ok::<_, IRTrace>(*slice))?[0];
                    ir.ir_mut().replace_single_input(op.id(), slice, input)?;
                } else if let Some(pad) = ir.parent_op::<PadAcrossDimension>(input)? {
                    let pad = ir.add_op(grandparents, Ok::<_, IRTrace>(*pad))?[0];
                    ir.ir_mut().replace_single_input(op.id(), pad, input)?;
                }
            }
        }

        ir.transform(EliminateUnusedOperations)
    }
}

// I don't want to write reduction kernels right now so scam it with matmul
rewriterule! {
    rulename ReduceToMatmul on ir
    rewrites op (output = [ReduceAcrossDimension] (input))
    {
        if output.dtype() == DType::F32 && output.reduction() == Reduction::Sum {
            let input = input.id();

            let (new_scalar, new_op) = if let Some(1) = output.inner().evaluate_constant() {
                let new_scalar = ir.add_scalar(1.0, output.dimen());
                let lhs = MatrixLayout { rows: 1.into(), cols: output.dimen(), col_mjr: true };
                let rhs = MatrixLayout { rows: output.dimen(), cols: output.outer(), col_mjr: true };
                (new_scalar, Matmul::new(DType::F32, 1, lhs, rhs)?)
            } else {
                let new_scalar = ir.add_scalar(1.0, output.outer() * output.dimen());
                let lhs = MatrixLayout { rows: 1.into(), cols: output.dimen(), col_mjr: true };
                let rhs = MatrixLayout { rows: output.dimen(), cols: output.inner(), col_mjr: false };
                (new_scalar, Matmul::new(DType::F32, output.outer(), lhs, rhs)?)
            };

            ir.replace_operation(op.id(), [new_scalar, input], new_op)?;
            return Ok(true);
        }
    }
}

// Rewrite Mx1 @ 1xN to broadcast and pointwise multiplication
rewriterule! {
    rulename MatmulToBroadcastMul on ir
    rewrites op (output = [Matmul] (lhs) (rhs))
    {
        if output.lhs.cols == Size::constant(1) {
            let m = output.lhs.rows;
            let n = output.rhs.cols;

            let lhs = lhs.id();
            let rhs = rhs.id();
            let lhs = ir.add_broadcast(lhs, [m], 0, n)?;
            let rhs = ir.add_broadcast(rhs, [n, 1.into()], 1, m)?;
            let ty = ir.get_node(lhs)?.ty();
            let new_op = CABinaryOp::new(ty, CABinary::Mul);

            ir.replace_operation(op.id(), [lhs, rhs], new_op)?;
            return Ok(true);
        }
    }
}

static REDUCTION_SRC: &str = "
extern \"C\" __global__ void kernel(int size, const float* input, float* output) {
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < (OUTER)) {
        float reduction = input[INNER * tid];

        for (int i = 1; i < INNER; i++) {
            reduction = FUNC(reduction, input[INNER * tid + i]);
        }

        output[tid] = reduction;
    }
}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodegenReduction;

impl IRTransform for CodegenReduction {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            if let Some(reduction) = op.data().downcast::<ReduceAcrossDimension>()
                && reduction.reduction() != Reduction::Sum
                && reduction.inner() == 1.into()
                && let Some(dimen) = reduction.dimen().evaluate_constant()
            {
                let outer = reduction.outer();
                let mut outer_str = format!("{}", outer.factor());
                for _ in 0..outer.var_power() {
                    outer_str += " * size";
                }

                let src = REDUCTION_SRC
                    .replace(
                        "FUNC",
                        match reduction.reduction() {
                            Reduction::Max => "max",
                            Reduction::Min => "min",
                            _ => unimplemented!(),
                        },
                    )
                    .replace("INNER", &dimen.to_string())
                    .replace("OUTER", &outer_str);

                let outer = reduction.outer();
                let new = unsafe {
                    KernelSrc::new(
                        reduction.inputs(),
                        reduction.outputs(),
                        "kernel".to_string(),
                        src,
                        true,
                        vec![(0, true), (0, false)],
                        Default::default(),
                        Rc::new(move |s| {
                            let x = outer.evaluate(s).div_ceil(256) as u32;
                            Dim3 { x, y: 1, z: 1 }
                        }),
                        Rc::new(|_| 256),
                        Rc::new(|_| 0),
                    )
                };

                ir.replace_op(op.id(), AddOperation::new(op.inputs(), Ok::<_, IRTrace>(TensorOp(Rc::new(new)))))?;
            }
        }

        Ok(())
    }
}
