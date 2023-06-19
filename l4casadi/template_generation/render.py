import os
import sys
import pathlib
import urllib
import shutil

TERA_VERSION = "0.0.34"

PLATFORM2TERA = {
    "linux": "linux",
    "darwin": "osx",
    "win32": "windows"
}


def get_tera_exec_path():
    TERA_PATH = pathlib.Path(__file__).parent.resolve() / 'bin' / 't_renderer'
    if os.name == 'nt':
        TERA_PATH += '.exe'
    return TERA_PATH


def get_tera():
    tera_path = get_tera_exec_path()

    if os.path.exists(tera_path) and os.access(tera_path, os.X_OK):
        return tera_path

    repo_url = "https://github.com/acados/tera_renderer/releases"
    url = "{}/download/v{}/t_renderer-v{}-{}".format(
        repo_url, TERA_VERSION, TERA_VERSION, PLATFORM2TERA[sys.platform])

    manual_install = 'For manual installation follow these instructions:\n'
    manual_install += '1 Download binaries from {}\n'.format(url)
    manual_install += '2 Copy them in {}\n'.format(tera_path)
    manual_install += '3 Strip the version and platform from the binaries: '
    manual_install += 'as t_renderer-v0.0.34-X -> t_renderer)\n'
    manual_install += '4 Enable execution privilege on the file "t_renderer" with:\n'
    manual_install += '"chmod +x {}"\n\n'.format(tera_path)

    msg = "\n"
    msg += 'Tera template render executable not found, '
    msg += 'while looking in path:\n{}\n'.format(tera_path)
    msg += 'In order to be able to render the templates, '
    msg += 'you need to download the tera renderer binaries from:\n'
    msg += '{}\n\n'.format(repo_url)
    msg += 'Do you wish to set up Tera renderer automatically?\n'
    msg += 'y/N? (press y to download tera or any key for manual installation)\n'

    if input(msg) == 'y':
        print("Dowloading {}".format(url))
        with urllib.request.urlopen(url) as response, open(tera_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Successfully downloaded t_renderer.")
        os.chmod(tera_path, 0o755)
        return tera_path

    msg_cancel = "\nYou cancelled automatic download.\n\n"
    msg_cancel += manual_install
    msg_cancel += "Once installed re-run your script.\n\n"
    print(msg_cancel)

    sys.exit(1)


def render_template(in_file, out_file, output_dir, json_path, template_glob=None):

    file_path = os.path.dirname(os.path.abspath(__file__))
    if template_glob is None:
        template_glob = os.path.join(file_path, 'c_templates_tera', '**', '*')
    cwd = os.getcwd()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    tera_path = get_tera()

    # call tera as system cmd
    os_cmd = f"{tera_path} '{template_glob}' '{in_file}' '{json_path}' '{out_file}'"
    # Windows cmd.exe can not cope with '...', so use "..." instead:
    if os.name == 'nt':
        os_cmd = os_cmd.replace('\'', '\"')

    status = os.system(os_cmd)
    if (status != 0):
        raise Exception(f'Rendering of {in_file} failed!\n\nAttempted to execute OS command:\n{os_cmd}\n\n')

    os.chdir(cwd)