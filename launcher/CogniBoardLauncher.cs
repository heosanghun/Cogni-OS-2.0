using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("CogniBoard")]
[assembly: AssemblyDescription("Cogni-OS 2.0 Genesis local sovereign AI control")]
[assembly: AssemblyCompany("Cogni-OS")]
[assembly: AssemblyProduct("Cogni-OS 2.0 Genesis")]
[assembly: AssemblyVersion("0.4.1.0")]
[assembly: AssemblyFileVersion("0.4.1.0")]

internal static class CogniBoardLauncher
{
    private const string DefaultModel = @"C:\Project\cognios\gemma4-e4b-it";

    [STAThread]
    private static int Main()
    {
        try
        {
            string projectRoot = FindProjectRoot();
            string manifest = Path.Combine(
                projectRoot,
                "config",
                "gemma4-e4b-it.manifest.toml"
            );
            string model = Environment.GetEnvironmentVariable("COGNI_OS_MODEL_DIR");
            if (String.IsNullOrWhiteSpace(model))
            {
                model = DefaultModel;
            }

            projectRoot = RequireSafePath(projectRoot, "project root");
            manifest = RequireSafePath(manifest, "model manifest");
            model = RequireSafePath(model, "model directory");
            if (!File.Exists(manifest))
            {
                throw new FileNotFoundException("Model manifest not found.", manifest);
            }
            if (!Directory.Exists(model))
            {
                throw new DirectoryNotFoundException(
                    "Local model directory not found: " + model
                );
            }

            PythonCommand python = FindPython();
            RunPreflight(python, projectRoot);
            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = python.Executable,
                Arguments = JoinArguments(
                    python.Prefix,
                    "-m",
                    "cogni_demo.server",
                    "--model",
                    model,
                    "--manifest",
                    manifest
                ),
                WorkingDirectory = projectRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            SetOfflineEnvironment(start);
            Process launched = Process.Start(start);
            if (launched == null)
            {
                throw new InvalidOperationException("Python runtime did not start.");
            }
            if (launched.WaitForExit(1500))
            {
                throw new InvalidOperationException(
                    "CogniBoard backend exited during startup. Run "
                    + "Run-CogniOS-Demo.cmd to inspect the diagnostic log."
                );
            }
            return 0;
        }
        catch (Exception error)
        {
            MessageBox.Show(
                error.Message,
                "CogniBoard 실행 오류",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }
    }

    private static string FindProjectRoot()
    {
        string directory = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
        DirectoryInfo parentInfo = Directory.GetParent(
            directory.TrimEnd(Path.DirectorySeparatorChar)
        );
        string parent = parentInfo == null ? directory : parentInfo.FullName;
        string[] candidates =
        {
            directory,
            Path.Combine(directory, "Cogni-OS-2-Genesis-source"),
            parent,
        };
        foreach (string candidate in candidates)
        {
            if (File.Exists(Path.Combine(candidate, "cogni_demo", "server.py")))
            {
                return Path.GetFullPath(candidate);
            }
        }
        throw new DirectoryNotFoundException(
            "Cogni-OS project files were not found beside the launcher."
        );
    }

    private static PythonCommand FindPython()
    {
        string configured = Environment.GetEnvironmentVariable("COGNI_OS_PYTHON");
        if (!String.IsNullOrWhiteSpace(configured))
        {
            configured = RequireSafePath(configured, "COGNI_OS_PYTHON");
            if (!File.Exists(configured))
            {
                throw new FileNotFoundException(
                    "COGNI_OS_PYTHON does not point to a file.",
                    configured
                );
            }
            return new PythonCommand(configured, "");
        }

        foreach (string name in new[] { "pythonw.exe", "python.exe", "pyw.exe", "py.exe" })
        {
            string located = FindOnPath(name);
            if (located != null)
            {
                return new PythonCommand(
                    located,
                    name.StartsWith("py", StringComparison.OrdinalIgnoreCase)
                        && !name.StartsWith("python", StringComparison.OrdinalIgnoreCase)
                            ? "-3"
                            : ""
                );
            }
        }
        throw new FileNotFoundException(
            "Python 3.11+ was not found. Install the project runtime or set COGNI_OS_PYTHON."
        );
    }

    private static void RunPreflight(PythonCommand python, string projectRoot)
    {
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = python.Executable,
            Arguments = JoinArguments(
                python.Prefix,
                "-c",
                "import sys,torch,transformers;"
                + "from cogni_core.cts_policy import load_default_bounded_cts_controller;"
                + "assert sys.version_info >= (3,11);"
                + "assert torch.cuda.is_available();"
                + "load_default_bounded_cts_controller(device='cpu');"
                + "import cogni_demo.server"
            ),
            WorkingDirectory = projectRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardError = true,
        };
        SetOfflineEnvironment(start);
        using (Process process = Process.Start(start))
        {
            if (process == null)
            {
                throw new InvalidOperationException("Python preflight did not start.");
            }
            StringBuilder standardError = new StringBuilder();
            process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs args)
            {
                if (args.Data != null && standardError.Length < 1500)
                {
                    int remaining = 1500 - standardError.Length;
                    string line = args.Data.Length <= remaining
                        ? args.Data
                        : args.Data.Substring(0, remaining);
                    standardError.AppendLine(line);
                }
            };
            process.BeginErrorReadLine();
            if (!process.WaitForExit(30000))
            {
                process.Kill();
                process.WaitForExit();
                throw new TimeoutException("Python/CUDA preflight exceeded 30 seconds.");
            }
            process.WaitForExit();
            string errorText = standardError.ToString().Trim();
            if (process.ExitCode != 0)
            {
                string detail = errorText.Length == 0
                    ? "Python 3.11+, CUDA PyTorch, Transformers, or Cogni-OS is unavailable."
                    : errorText;
                if (detail.Length > 1500)
                {
                    detail = detail.Substring(0, 1500);
                }
                throw new InvalidOperationException("Runtime preflight failed:\n" + detail);
            }
        }
    }

    private static string FindOnPath(string fileName)
    {
        string path = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (string raw in path.Split(Path.PathSeparator))
        {
            string directory = raw.Trim().Trim('"');
            if (directory.Length == 0)
            {
                continue;
            }
            try
            {
                string candidate = Path.GetFullPath(Path.Combine(directory, fileName));
                if (File.Exists(candidate))
                {
                    return RequireSafePath(candidate, "Python runtime");
                }
            }
            catch (ArgumentException)
            {
                // Ignore malformed PATH entries and continue with the bounded list.
            }
            catch (NotSupportedException)
            {
                // Ignore malformed PATH entries and continue with the bounded list.
            }
        }
        return null;
    }

    private static string RequireSafePath(string value, string label)
    {
        if (
            String.IsNullOrWhiteSpace(value)
            || value.IndexOf('\0') >= 0
            || value.IndexOf('\r') >= 0
            || value.IndexOf('\n') >= 0
            || value.IndexOf('"') >= 0
        )
        {
            throw new ArgumentException(label + " contains unsupported characters.");
        }
        string full = Path.GetFullPath(value);
        if (
            full.StartsWith(@"\\", StringComparison.Ordinal)
            || full.StartsWith(@"\\?\", StringComparison.Ordinal)
            || full.StartsWith(@"\\.\", StringComparison.Ordinal)
        )
        {
            throw new ArgumentException(label + " must be on a local fixed volume.");
        }
        string root = Path.GetPathRoot(full);
        if (String.IsNullOrEmpty(root))
        {
            throw new ArgumentException(label + " has no local volume root.");
        }
        DriveInfo drive = new DriveInfo(root);
        if (drive.DriveType != DriveType.Fixed)
        {
            throw new ArgumentException(label + " must be on a local fixed volume.");
        }
        string current = File.Exists(full) ? Path.GetDirectoryName(full) : full;
        while (!String.IsNullOrEmpty(current) && Directory.Exists(current))
        {
            DirectoryInfo info = new DirectoryInfo(current);
            if ((info.Attributes & FileAttributes.ReparsePoint) != 0)
            {
                throw new ArgumentException(label + " crosses a reparse point.");
            }
            current = info.Parent == null ? null : info.Parent.FullName;
        }
        return full;
    }

    private static string JoinArguments(params string[] values)
    {
        List<string> result = new List<string>();
        foreach (string value in values)
        {
            if (String.IsNullOrEmpty(value))
            {
                continue;
            }
            if (value.IndexOf('"') >= 0 || value.IndexOf('\0') >= 0)
            {
                throw new ArgumentException("Launcher argument contains unsupported characters.");
            }
            result.Add(QuoteArgument(value));
        }
        return String.Join(" ", result);
    }

    private static string QuoteArgument(string value)
    {
        StringBuilder result = new StringBuilder();
        result.Append('"');
        int backslashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                backslashes++;
                continue;
            }
            if (character == '"')
            {
                result.Append('\\', backslashes * 2 + 1);
                result.Append('"');
                backslashes = 0;
                continue;
            }
            result.Append('\\', backslashes);
            backslashes = 0;
            result.Append(character);
        }
        result.Append('\\', backslashes * 2);
        result.Append('"');
        return result.ToString();
    }

    private static void SetOfflineEnvironment(ProcessStartInfo start)
    {
        start.EnvironmentVariables["HF_HUB_OFFLINE"] = "1";
        start.EnvironmentVariables["HF_HUB_DISABLE_TELEMETRY"] = "1";
        start.EnvironmentVariables["TRANSFORMERS_OFFLINE"] = "1";
        start.EnvironmentVariables["HF_DATASETS_OFFLINE"] = "1";
        start.EnvironmentVariables["WANDB_MODE"] = "offline";
        start.EnvironmentVariables["TOKENIZERS_PARALLELISM"] = "false";
        start.EnvironmentVariables["PYTHONUTF8"] = "1";
    }

    private sealed class PythonCommand
    {
        internal PythonCommand(string executable, string prefix)
        {
            Executable = executable;
            Prefix = prefix;
        }

        internal string Executable { get; private set; }
        internal string Prefix { get; private set; }
    }
}
