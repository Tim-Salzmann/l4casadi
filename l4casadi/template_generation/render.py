import os
import jinja2


def render_casadi_c_template(variables, out_file):
    file_path = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(file_path, 'templates')

    render_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_path),
        autoescape=jinja2.select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True
    )

    template = render_env.get_template("casadi_function.in.cpp")

    with open(out_file, "w") as f:
        f.write(template.render(variables))
