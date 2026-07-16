use std::{cell::RefCell, collections::BTreeSet, ffi::c_void, fmt, rc::Rc, sync::Arc};

use bullet_compiler::tensor::{OpType, TType};

use crate::{
    buffer::{Buffer, SyncOnDrop, SyncOnValue},
    runtime::{Device, Dim3, Gpu, Kernel, Module, Stream},
};

#[derive(Clone)]
pub struct KernelSrc {
    pub(crate) inputs: Vec<TType>,
    pub(crate) outputs: Vec<TType>,
    pub(crate) name: String,
    pub(crate) source: String,
    pub(crate) requires_var_size_arg: bool,
    pub(crate) arg_order: Vec<(usize, bool)>,
    pub(crate) requires_zero: BTreeSet<usize>,
    pub(crate) gdim: Rc<dyn Fn(usize) -> Dim3>,
    pub(crate) bdim: Rc<dyn Fn(usize) -> u32>,
    pub(crate) smem: Rc<dyn Fn(usize) -> u32>,
}

impl fmt::Debug for KernelSrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu.kernel.source")
    }
}

impl KernelSrc {
    /// ### Safety
    ///
    /// I solemnly swear that as long as the passed input and output
    /// tensors to the compiled function have the correct TType and
    /// the variable size is passed correctly, then this kernel will
    /// not invoke UB.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        inputs: Vec<TType>,
        outputs: Vec<TType>,
        name: String,
        source: String,
        requires_var_size_arg: bool,
        arg_order: Vec<(usize, bool)>,
        requires_zero: BTreeSet<usize>,
        gdim: Rc<dyn Fn(usize) -> Dim3>,
        bdim: Rc<dyn Fn(usize) -> u32>,
        smem: Rc<dyn Fn(usize) -> u32>,
    ) -> Self {
        assert_eq!(arg_order.len(), inputs.len() + outputs.len());
        assert_eq!(
            inputs.len(),
            arg_order.iter().filter_map(|(idx, input)| input.then_some(*idx)).collect::<BTreeSet<_>>().len()
        );
        assert_eq!(
            outputs.len(),
            arg_order.iter().filter_map(|(idx, input)| (!input).then_some(*idx)).collect::<BTreeSet<_>>().len()
        );

        Self { inputs, outputs, name, source, requires_var_size_arg, arg_order, requires_zero, gdim, bdim, smem }
    }

    pub fn compile<G: Gpu>(&self, device: Arc<Device<G>>) -> Result<CompiledKernel<G>, G::Error> {
        let kernel = Module::new(device, &self.source)?.get_kernel(&self.name)?;

        Ok(CompiledKernel {
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            kernel,
            requires_var_size_arg: self.requires_var_size_arg,
            arg_order: self.arg_order.clone(),
            requires_zero: self.requires_zero.clone(),
            gdim: self.gdim.clone(),
            bdim: self.bdim.clone(),
            smem: self.smem.clone(),
            scratch: RefCell::new(CompiledKernelScratch::new(
                self.inputs.len(),
                self.outputs.len(),
                self.arg_order.len() + usize::from(self.requires_var_size_arg),
                self.arg_order.len(),
            )),
        })
    }
}

impl OpType for KernelSrc {
    fn opname(&self) -> String {
        format!("gpu.rtc.{}", self.name)
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.clone()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.clone()
    }
}

pub struct CompiledKernel<G: Gpu> {
    pub(crate) inputs: Vec<TType>,
    pub(crate) outputs: Vec<TType>,
    pub(crate) kernel: Kernel<G>,
    pub(crate) requires_var_size_arg: bool,
    pub(crate) arg_order: Vec<(usize, bool)>,
    pub(crate) requires_zero: BTreeSet<usize>,
    pub(crate) gdim: Rc<dyn Fn(usize) -> Dim3>,
    pub(crate) bdim: Rc<dyn Fn(usize) -> u32>,
    pub(crate) smem: Rc<dyn Fn(usize) -> u32>,
    scratch: RefCell<CompiledKernelScratch<G>>,
}

struct CompiledKernelScratch<G: Gpu> {
    input_ptrs: Vec<G::DevicePtr>,
    output_ptrs: Vec<G::DevicePtr>,
    args: Vec<*mut c_void>,
    ptr_values: Vec<G::DevicePtr>,
}

impl<G: Gpu> CompiledKernelScratch<G> {
    fn new(input_count: usize, output_count: usize, max_args: usize, ptr_count: usize) -> Self {
        Self {
            input_ptrs: Vec::with_capacity(input_count),
            output_ptrs: Vec::with_capacity(output_count),
            args: Vec::with_capacity(max_args),
            ptr_values: Vec::with_capacity(ptr_count),
        }
    }

    fn prepare(&mut self) {
        self.input_ptrs.clear();
        self.output_ptrs.clear();
        self.args.clear();
        self.ptr_values.clear();
    }
}

impl<G: Gpu> fmt::Debug for CompiledKernel<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu.kernel.compiled")
    }
}

impl<G: Gpu> CompiledKernel<G> {
    /// ### Safety
    ///
    /// I solemnly swear that as long as the passed input and output
    /// tensors to the kernel have the correct TType and the variable
    /// size is passed correctly, then this kernel will not invoke UB.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        inputs: Vec<TType>,
        outputs: Vec<TType>,
        kernel: Kernel<G>,
        requires_var_size_arg: bool,
        arg_order: Vec<(usize, bool)>,
        requires_zero: BTreeSet<usize>,
        gdim: Rc<dyn Fn(usize) -> Dim3>,
        bdim: Rc<dyn Fn(usize) -> u32>,
        smem: Rc<dyn Fn(usize) -> u32>,
    ) -> Self {
        assert_eq!(arg_order.len(), inputs.len() + outputs.len());
        assert_eq!(
            inputs.len(),
            arg_order.iter().filter_map(|(idx, input)| input.then_some(*idx)).collect::<BTreeSet<_>>().len()
        );
        assert_eq!(
            outputs.len(),
            arg_order.iter().filter_map(|(idx, input)| (!input).then_some(*idx)).collect::<BTreeSet<_>>().len()
        );

        let input_count = inputs.len();
        let output_count = outputs.len();
        let max_args = arg_order.len() + usize::from(requires_var_size_arg);
        let ptr_count = arg_order.len();

        Self {
            inputs,
            outputs,
            kernel,
            requires_var_size_arg,
            arg_order,
            requires_zero,
            gdim,
            bdim,
            smem,
            scratch: RefCell::new(CompiledKernelScratch::new(input_count, output_count, max_args, ptr_count)),
        }
    }

    pub fn execute(
        &self,
        stream: Arc<Stream<G>>,
        inputs: Vec<Arc<Buffer<G>>>,
        outputs: Vec<Arc<Buffer<G>>>,
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
        self.execute_slices(stream, &inputs, &outputs)
    }

    pub fn execute_slices(
        &self,
        stream: Arc<Stream<G>>,
        inputs: &[Arc<Buffer<G>>],
        outputs: &[Arc<Buffer<G>>],
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
        self.execute_iter(stream, inputs.iter(), outputs.iter(), inputs.len(), outputs.len())
    }

    pub fn execute_ref_slices(
        &self,
        stream: Arc<Stream<G>>,
        inputs: &[&Arc<Buffer<G>>],
        outputs: &[&Arc<Buffer<G>>],
    ) -> Result<SyncOnValue<G, &Self>, G::Error> {
        self.execute_iter(stream, inputs.iter().copied(), outputs.iter().copied(), inputs.len(), outputs.len())
    }

    fn execute_iter<'a, 'b>(
        &'a self,
        stream: Arc<Stream<G>>,
        inputs: impl IntoIterator<Item = &'b Arc<Buffer<G>>>,
        outputs: impl IntoIterator<Item = &'b Arc<Buffer<G>>>,
        input_len: usize,
        output_len: usize,
    ) -> Result<SyncOnValue<G, &'a Self>, G::Error>
    where
        G: 'b,
    {
        if input_len != self.inputs.len() || output_len != self.outputs.len() {
            return Err("Mismatched number of inputs/outputs!".to_string().into());
        }

        let mut sync = SyncOnDrop::with_capacity(stream.clone(), input_len + output_len);
        let mut scratch = self.scratch.borrow_mut();
        scratch.prepare();

        let mut var_size = None;

        for (input, &ttype) in inputs.into_iter().zip(&self.inputs) {
            let guard = input.acquire(stream.clone())?;
            if guard.dtype() != ttype.dtype() {
                return Err("Mismatched dtypes!".to_string().into());
            }

            let concrete_size = guard.size();
            if let Some(var) = ttype.size().get_var_size(concrete_size) {
                match var_size {
                    None => var_size = Some(var),
                    Some(old_var) if old_var == var => {}
                    Some(old_var) => {
                        return Err(format!("Mismatching batch sizes in inputs: {old_var} != {var}").into());
                    }
                }
            } else if ttype.size().evaluate_constant().unwrap() != concrete_size {
                return Err("Mismatched sizes!".to_string().into());
            }

            scratch.input_ptrs.push(guard.ptr());
            sync.attach(guard)?;
        }

        for (output, &ttype) in outputs.into_iter().zip(&self.outputs) {
            let guard = output.acquire(stream.clone())?;
            if guard.dtype() != ttype.dtype() {
                return Err("Mismatched dtypes!".to_string().into());
            }

            let concrete_size = guard.size();
            if let Some(var) = ttype.size().get_var_size(concrete_size) {
                match var_size {
                    None => var_size = Some(var),
                    Some(old_var) if old_var == var => {}
                    Some(old_var) => {
                        return Err(format!("Mismatching batch sizes in inputs: {old_var} != {var}").into());
                    }
                }
            } else if ttype.size().evaluate_constant().unwrap() != concrete_size {
                return Err("Mismatched sizes!".to_string().into());
            }

            scratch.output_ptrs.push(guard.ptr());
            sync.attach(guard)?;
        }

        let var = var_size.unwrap_or(1);

        let size = var as i32;
        if self.requires_var_size_arg {
            scratch.args.push((&size as *const i32).cast_mut().cast());
        }

        for &(index, is_input) in &self.arg_order {
            let ptr = if is_input { scratch.input_ptrs[index] } else { scratch.output_ptrs[index] };
            scratch.ptr_values.push(ptr);
            let value_idx = scratch.ptr_values.len() - 1;
            let arg_ptr = (&scratch.ptr_values[value_idx] as *const G::DevicePtr).cast_mut().cast();
            scratch.args.push(arg_ptr);
        }

        unsafe {
            if !self.requires_zero.is_empty() {
                unimplemented!();
            }

            self.kernel.launch(
                &stream,
                (self.gdim)(var),
                (self.bdim)(var),
                scratch.args.as_ptr().cast_mut().cast(),
                (self.smem)(var),
            )?;
        }

        Ok(SyncOnValue::new(sync, self))
    }
}

impl<G: Gpu> OpType for CompiledKernel<G> {
    fn opname(&self) -> String {
        "gpu.kernel.compiled".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.clone()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.clone()
    }
}
