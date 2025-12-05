#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys

from cutile_from_aten import compile_module_to_cutile


def run_test(name: str, module: nn.Module, inputs: tuple, ref_fn=None, rtol=1e-5, atol=1e-5) -> bool:
    print(f"\n{'-'*60}")
    print(f"TEST: {name}")
    print(f"{'-'*60}")
    
    try:
        compiled_fn = compile_module_to_cutile(module, *inputs, debug=False)
        result = compiled_fn(*inputs)
        
        if ref_fn is None:
            ref = module(*inputs)
        else:
            ref = ref_fn(*inputs)
        
        if torch.allclose(result, ref, rtol=rtol, atol=atol):
            print(f"PASSED: {name}")
            return True
        else:
            max_diff = (result - ref).abs().max().item()
            print(f"FAILED: {name}")
            print(f"  Max diff: {max_diff}")
            return False
            
    except Exception as e:
        print(f"ERROR: {name}")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False


def make_tensors(*shapes, device="cuda", dtype=torch.float32):
    return tuple(torch.randn(s, device=device, dtype=dtype) for s in shapes)


def test_binary_ops():
    device = "cuda"
    size = 4096
    a, b = make_tensors(size, size, device=device)
    
    results = []
    
    class Add(nn.Module):
        def forward(self, x, y): return x + y
    results.append(run_test("aten.add.Tensor", Add(), (a, b)))
    
    class Sub(nn.Module):
        def forward(self, x, y): return x - y
    results.append(run_test("aten.sub.Tensor", Sub(), (a, b)))
    
    class Mul(nn.Module):
        def forward(self, x, y): return x * y
    results.append(run_test("aten.mul.Tensor", Mul(), (a, b)))
    
    class Div(nn.Module):
        def forward(self, x, y): return x / y
    results.append(run_test("aten.div.Tensor", Div(), (a, b)))
    
    return all(results)


def test_unary_ops():
    device = "cuda"
    size = 4096
    
    pos = torch.rand(size, device=device) + 0.1
    any_val = torch.randn(size, device=device)
    small = torch.randn(size, device=device) * 0.5
    
    results = []
    
    class Neg(nn.Module):
        def forward(self, x): return -x
    results.append(run_test("aten.neg.default", Neg(), (any_val,)))
    
    class Exp(nn.Module):
        def forward(self, x): return torch.exp(x)
    results.append(run_test("aten.exp.default", Exp(), (small,)))
    
    class Log(nn.Module):
        def forward(self, x): return torch.log(x)
    results.append(run_test("aten.log.default", Log(), (pos,)))
    
    class Sqrt(nn.Module):
        def forward(self, x): return torch.sqrt(x)
    results.append(run_test("aten.sqrt.default", Sqrt(), (pos,)))
    
    class Tanh(nn.Module):
        def forward(self, x): return torch.tanh(x)
    results.append(run_test("aten.tanh.default", Tanh(), (any_val,)))
    
    class Cos(nn.Module):
        def forward(self, x): return torch.cos(x)
    results.append(run_test("aten.cos.default", Cos(), (any_val,)))
    
    class Sin(nn.Module):
        def forward(self, x): return torch.sin(x)
    results.append(run_test("aten.sin.default", Sin(), (any_val,)))
    
    return all(results)


def test_chained_ops():
    device = "cuda"
    size = 4096
    a, b, c = make_tensors(size, size, size, device=device)
    pos = torch.rand(size, device=device) + 0.1
    
    results = []
    
    class AddMul(nn.Module):
        def forward(self, a, b, c): return (a + b) * c
    results.append(run_test("(a + b) * c", AddMul(), (a, b, c)))
    
    class MulAdd(nn.Module):
        def forward(self, a, b, c): return a * b + c
    results.append(run_test("a * b + c", MulAdd(), (a, b, c)))
    
    class Complex1(nn.Module):
        def forward(self, a, b, c): 
            return torch.tanh((a + b) * c) - torch.exp(a * 0.1)
    results.append(run_test("tanh((a+b)*c) - exp(a*0.1)", Complex1(), (a, b, c)))
    
    class Complex2(nn.Module):
        def forward(self, a, b, c):
            return torch.sqrt(a) * torch.cos(b) + torch.sin(c)
    results.append(run_test("sqrt(a) * cos(b) + sin(c)", Complex2(), (pos, b, c)))
    
    return all(results)


def test_larger_sizes():
    device = "cuda"
    
    class AddMul(nn.Module):
        def forward(self, a, b, c): return (a + b) * c
    
    results = []
    
    for size in [1024, 1024*1024, 4*1024*1024]:
        a, b, c = make_tensors(size, size, size, device=device)
        results.append(run_test(f"size={size}", AddMul(), (a, b, c)))
    
    return all(results)


def run_all_tests():
    
    all_results = []
    
    all_results.append(("Binary Ops", test_binary_ops()))
    all_results.append(("Unary Ops", test_unary_ops()))
    all_results.append(("Chained Ops", test_chained_ops()))
    all_results.append(("Larger Sizes", test_larger_sizes()))
    
    
    passed = sum(1 for _, r in all_results if r)
    total = len(all_results)
    
    for name, result in all_results:
        status = "PASSED" if result else "FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
