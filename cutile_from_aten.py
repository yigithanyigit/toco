#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from torch.export import export
from typing import Dict, List, Any, Optional
import torch.fx as fx

try:
    from .hints import pointwise_block_size, cdiv, CUTILE_MAX_BLOCK
except ImportError:
    from hints import pointwise_block_size, cdiv, CUTILE_MAX_BLOCK


def _is_debug_enabled() -> bool:
    val = os.environ.get("DEBUG", "").lower()
    return val in ("1", "true", "yes")


DEFAULT_BLOCK_SIZE = 1024


#aten_op_name -> (format_string, num_args)
ATEN_OP_MAPPINGS = {
    # binary ops (aten.op.Tensor)
    "aten.add.Tensor": ("({a} + {b})", 2),
    "aten.sub.Tensor": ("({a} - {b})", 2),
    "aten.mul.Tensor": ("({a} * {b})", 2),
    "aten.div.Tensor": ("({a} / {b})", 2),
    "aten.pow.Tensor_Tensor": ("ct.pow({a}, {b})", 2),
    "aten.maximum.default": ("ct.maximum({a}, {b})", 2),
    "aten.minimum.default": ("ct.minimum({a}, {b})", 2),
    
    # unary ops (aten.op.default)
    "aten.neg.default": ("(-{x})", 1),
    "aten.abs.default": ("ct.abs({x})", 1),
    "aten.exp.default": ("ct.exp({x})", 1),
    "aten.exp2.default": ("ct.exp2({x})", 1),
    "aten.log.default": ("ct.log({x})", 1),
    "aten.log2.default": ("ct.log2({x})", 1),
    "aten.sqrt.default": ("ct.sqrt({x})", 1),
    "aten.rsqrt.default": ("ct.rsqrt({x})", 1),
    "aten.cos.default": ("ct.cos({x})", 1),
    "aten.sin.default": ("ct.sin({x})", 1),
    "aten.tan.default": ("ct.tan({x})", 1),
    "aten.tanh.default": ("ct.tanh({x})", 1),
    "aten.cosh.default": ("ct.cosh({x})", 1),
    "aten.sinh.default": ("ct.sinh({x})", 1),
    "aten.floor.default": ("ct.floor({x})", 1),
    "aten.ceil.default": ("ct.ceil({x})", 1),
    "aten.sigmoid.default": ("(1.0 / (1.0 + ct.exp(-{x})))", 1),  # sigmoid = 1/(1+exp(-x))
    "aten.relu.default": ("ct.maximum({x}, 0.0)", 1),
}


class CuTileAtenCodeGen:
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE):
        self.block_size = block_size
        self.var_counter = 0
        self.var_map: Dict[fx.Node, str] = {}
        self.loads: List[str] = []
        self.compute: List[str] = []
        self.input_args: List[str] = []
        self.output_arg: str = "Out"
        
    def _fresh_var(self) -> str:
        name = f"tmp{self.var_counter}"
        self.var_counter += 1
        return name
    
    def _get_node_value(self, node: fx.Node) -> str:
        if node in self.var_map:
            return self.var_map[node]
        raise ValueError(f"Unknown node: {node}")
    
    def _get_arg_value(self, arg: Any) -> str:
        if isinstance(arg, fx.Node):
            return self._get_node_value(arg)
        elif isinstance(arg, (int, float)):
            return repr(arg)
        elif isinstance(arg, bool):
            return "True" if arg else "False"
        else:
            return str(arg)
    
    def _get_aten_op_name(self, target) -> str:
        return str(target)
    
    def _process_placeholder(self, node: fx.Node, idx: int) -> None:
        arg_name = node.name if node.name else f"in{idx}"
        self.input_args.append(arg_name)
        
        var = self._fresh_var()
        self.var_map[node] = var
        self.loads.append(f"{var} = ct.gather({arg_name}, global_idx)")
    
    def _process_call_function(self, node: fx.Node) -> None:
        target = node.target
        op_name = self._get_aten_op_name(target)
        
        if op_name not in ATEN_OP_MAPPINGS:
            raise NotImplementedError(f"ATen operation '{op_name}' not supported. Target: {target}")
        
        format_str, num_args = ATEN_OP_MAPPINGS[op_name]
        args = node.args
        
        if num_args == 1:
            # unary
            x_val = self._get_arg_value(args[0])
            expr = format_str.format(x=x_val)
        elif num_args == 2:
            # binary
            a_val = self._get_arg_value(args[0])
            b_val = self._get_arg_value(args[1])
            expr = format_str.format(a=a_val, b=b_val)
        else:
            raise ValueError(f"Unsupported number of args: {num_args}")
        
        var = self._fresh_var()
        self.var_map[node] = var
        self.compute.append(f"{var} = {expr}")
    
    def _process_output(self, node: fx.Node) -> str:
        if len(node.args) == 1:
            output_arg = node.args[0]
            if isinstance(output_arg, fx.Node):
                return self._get_node_value(output_arg)
            elif isinstance(output_arg, (tuple, list)) and len(output_arg) == 1:
                return self._get_node_value(output_arg[0])
        raise ValueError(f"Unexpected output format: {node.args}")
    
    def generate(self, exported_program, kernel_name: str = "cutile_kernel") -> str:
        # self.var_counter = 0
        # self.var_map = {}
        # self.loads = []
        # self.compute = []
        # self.input_args = []
        
        graph_module = exported_program.graph_module
        
        placeholder_idx = 0
        output_var = None
        
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                self._process_placeholder(node, placeholder_idx)
                placeholder_idx += 1
            elif node.op == "call_function":
                self._process_call_function(node)
            elif node.op == "output":
                output_var = self._process_output(node)
        
        if output_var is None:
            raise ValueError("No output node found in graph")
        
        return self._build_kernel_code(kernel_name, output_var)
    
    def _build_kernel_code(self, kernel_name: str, output_var: str) -> str:
        all_args = self.input_args + [self.output_arg]
        args_str = ", ".join(all_args)
        
        def indent(lines: List[str], level: int = 1) -> str:
            prefix = "    " * level
            return "\n".join(prefix + line for line in lines)
        
        body_lines = [
            "# Get block ID",
            "block_id = ct.bid(0)",
            "",
            "# Create global indices",
            "local_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)",
            "global_idx = block_id * BLOCK_SIZE + local_idx",
            "",
            "# Load inputs",
        ]
        body_lines.extend(self.loads)
        body_lines.append("")
        body_lines.append("# Compute")
        body_lines.extend(self.compute)
        body_lines.append("")
        body_lines.append("# Store output")
        body_lines.append(f"ct.scatter({self.output_arg}, global_idx, {output_var})")
        
        kernel_code = f'''import cuda.tile as ct

BLOCK_SIZE = {self.block_size}

@ct.kernel
def {kernel_name}({args_str}):
{indent(body_lines)}


def launch_{kernel_name}({args_str}, stream=None):
    import torch
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    
    numel = {self.input_args[0]}.numel()
    num_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    
    ct.launch(stream, grid, {kernel_name}, ({args_str}))
'''
        return kernel_code


def compile_module_to_cutile(
    module: nn.Module, 
    *example_inputs, 
    debug: bool = False,
    block_size: Optional[int] = None,
):
    import tempfile
    import importlib.util
    from pathlib import Path
    
    debug = debug or _is_debug_enabled()
    
    exported = export(module, example_inputs)
    
    if debug:
        print("------ Exported Graph ------")
        print(exported.graph_module)
        print()
    
    if block_size is None:
        first_input = example_inputs[0]
        numel = first_input.numel()
        dtype = first_input.dtype
        block_size = pointwise_block_size(numel, dtype)
        if debug:
            print(f"------ Heuristics ------")
            print(f"numel={numel}, dtype={dtype} → block_size={block_size}")
            print()
    
    codegen = CuTileAtenCodeGen(block_size=block_size)
    kernel_name = "cutile_aten_kernel"
    code = codegen.generate(exported, kernel_name=kernel_name)
    
    if debug:
        print("------ Generated cuTile Code ------")
        print(code)
        print()
    
    temp_dir = Path(tempfile.gettempdir()) / "cutile_aten_kernels"
    temp_dir.mkdir(exist_ok=True)
    
    module_path = temp_dir / f"{kernel_name}.py"
    module_path.write_text(code)
    
    if debug:
        print(f"kernel exported to: {module_path}")
    
    spec = importlib.util.spec_from_file_location(kernel_name, module_path)
    compiled_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled_module)
    
    kernel_fn = getattr(compiled_module, kernel_name)
    launch_fn = getattr(compiled_module, f"launch_{kernel_name}")
    
    def run(*args):
        first_arg = args[0]
        out = torch.empty_like(first_arg)
        launch_fn(*args, out)
        torch.cuda.synchronize()
        return out
    
    return run


if __name__ == "__main__":
    
    class TestModule(nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    
    device = "cuda"
    a = torch.randn(4096, device=device)
    b = torch.randn(4096, device=device)
    c = torch.randn(4096, device=device)
    
    compiled_fn = compile_module_to_cutile(TestModule(), a, b, c, debug=True)
    result = compiled_fn(a, b, c)
    
    ref = (a + b) * c
    
    if torch.allclose(result, ref, rtol=1e-5, atol=1e-5):
        print("\n Test passed,  cuTile output matches PyTorch reference.")
    else:
        print(f"\n Test failed! Max diff: {(result - ref).abs().max().item()}")
