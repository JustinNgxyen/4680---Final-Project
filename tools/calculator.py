import ast
import operator

# allowed operations
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def eval_expr(expr: str) -> str:
    def _eval(node):
        if isinstance(node, ast.BinOp):
            return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.Num):
            return node.n
        else:
            raise ValueError("Invalid expression")

    tree = ast.parse(expr, mode="eval")
    return str(_eval(tree.body))