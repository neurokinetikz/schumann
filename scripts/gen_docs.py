import os
import ast
import glob

def get_args_string(args):
    arg_list = []
    # Positional args
    for arg in args.args:
        arg_str = arg.arg
        if arg.annotation:
            # We skip complex annotation parsing for brevity in some cases, 
            # but unparse is best if available (Python 3.9+)
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except:
                pass 
        arg_list.append(arg_str)
    
    # Defaults (they align with the last n args)
    if args.defaults:
        offset = len(args.args) - len(args.defaults)
        for i, default in enumerate(args.defaults):
            try:
                base = arg_list[offset + i]
                arg_list[offset + i] = f"{base}={ast.unparse(default)}"
            except:
                pass

    if args.vararg:
        arg_list.append(f"*{args.vararg.arg}")
    
    if args.kwonlyargs:
        # If there are kwonlyargs but no vararg, we need a bare *
        if not args.vararg: 
            # This is implicit in python signature but explicit in display usually
            pass # simplified
        for i, kwarg in enumerate(args.kwonlyargs):
            kw_str = kwarg.arg
            if kwarg.annotation:
                 try:
                    kw_str += f": {ast.unparse(kwarg.annotation)}"
                 except:
                     pass
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                try:
                    kw_str += f"={ast.unparse(args.kw_defaults[i])}"
                except:
                    pass
            arg_list.append(kw_str)

    if args.kwarg:
        arg_list.append(f"**{args.kwarg.arg}")

    return ", ".join(arg_list)

def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"## File: `{filename}`\n")
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception as e:
        print(f"> Error parsing file: {e}\n")
        return

    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        print(f"{module_doc.strip()}\n")
    
    # Iterate nodes
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            print_function(node, level=3)
        elif isinstance(node, ast.ClassDef):
            print_class(node)

def print_function(node, level=3):
    header_prefix = "#" * level
    args = get_args_string(node.args)
    
    # Return type
    ret_ann = ""
    if node.returns:
        try:
            ret_ann = f" -> {ast.unparse(node.returns)}"
        except:
            pass
            
    print(f"{header_prefix} function `{node.name}({args}){ret_ann}`")
    doc = ast.get_docstring(node)
    if doc:
        # Indent docstring slightly or just print as blockquote
        print(f"> {doc.strip().replace(chr(10), chr(10)+'> ')}\n")
    else:
        print("\n")

def print_class(node):
    print(f"### class `{node.name}`")
    doc = ast.get_docstring(node)
    if doc:
        print(f"> {doc.strip().replace(chr(10), chr(10)+'> ')}\n")
    else:
        print("\n")
        
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            print_function(item, level=4)

def main():
    print("# Library Reference Documentation\n")
    lib_files = sorted(glob.glob("lib/*.py"))
    for f in lib_files:
        process_file(f)
        print("---\n")

if __name__ == "__main__":
    main()
