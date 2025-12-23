//! Shell command toolkit.
//!
//! Provides the ability to execute shell commands with safety restrictions.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;

use crate::error::{AgnoError, Result};
use crate::tool::{Tool, ToolRegistry};

/// Configuration for Shell tools
#[derive(Clone)]
pub struct ShellConfig {
    /// Base directory for command execution
    pub base_dir: Option<PathBuf>,
    /// Maximum number of output lines to return
    pub max_output_lines: usize,
    /// Command timeout in seconds
    pub timeout_secs: u64,
    /// List of blocked commands for safety
    pub blocked_commands: Vec<String>,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            base_dir: None,
            max_output_lines: 100,
            timeout_secs: 30,
            blocked_commands: vec![
                "rm -rf /".into(),
                "rm -rf /*".into(),
                "mkfs".into(),
                "dd if=".into(),
                ":(){:|:&};:".into(), // fork bomb
            ],
        }
    }
}

/// Create a Shell toolkit
pub fn shell_toolkit(config: ShellConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(RunShellCommandTool { config });
    registry
}

struct RunShellCommandTool {
    config: ShellConfig,
}

#[async_trait]
impl Tool for RunShellCommandTool {
    fn name(&self) -> &str {
        "run_shell_command"
    }

    fn description(&self) -> &str {
        "Execute a shell command. Expects {\"command\": string} or {\"args\": [string array]}. Returns stdout or error."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        // Get command either as a single string or args array
        let (program, args): (String, Vec<String>) = if let Some(cmd) = input.get("command").and_then(Value::as_str) {
            // Parse command string
            #[cfg(unix)]
            {
                ("sh".into(), vec!["-c".into(), cmd.into()])
            }
            #[cfg(windows)]
            {
                ("cmd".into(), vec!["/C".into(), cmd.into()])
            }
        } else if let Some(args) = input.get("args").and_then(Value::as_array) {
            let args: Vec<String> = args
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect();
            if args.is_empty() {
                return Err(AgnoError::Protocol(
                    "empty `args` for run_shell_command".into(),
                ));
            }
            let program = args[0].clone();
            let remaining = args.into_iter().skip(1).collect();
            (program, remaining)
        } else {
            return Err(AgnoError::Protocol(
                "missing `command` or `args` for run_shell_command".into(),
            ));
        };

        // Safety check
        let full_command = format!("{} {}", program, args.join(" "));
        for blocked in &self.config.blocked_commands {
            if full_command.contains(blocked) {
                return Ok(json!({
                    "error": format!("Command blocked for safety: contains '{}'", blocked),
                    "exit_code": -1
                }));
            }
        }

        // Build command
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(ref base_dir) = self.config.base_dir {
            cmd.current_dir(base_dir);
        }

        // Execute with timeout
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.timeout_secs),
            cmd.output(),
        )
        .await
        .map_err(|_| AgnoError::ToolInvocation {
            name: "run_shell_command".into(),
            source: "Command timed out".into(),
        })?
        .map_err(|e| AgnoError::ToolInvocation {
            name: "run_shell_command".into(),
            source: Box::new(e),
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Limit output lines
        let stdout_lines: Vec<&str> = stdout.lines().collect();
        let truncated_stdout: String = if stdout_lines.len() > self.config.max_output_lines {
            stdout_lines[stdout_lines.len() - self.config.max_output_lines..]
                .join("\n")
        } else {
            stdout.to_string()
        };

        if output.status.success() {
            Ok(json!({
                "stdout": truncated_stdout,
                "exit_code": output.status.code().unwrap_or(0)
            }))
        } else {
            Ok(json!({
                "error": stderr.to_string(),
                "stdout": truncated_stdout,
                "exit_code": output.status.code().unwrap_or(-1)
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_command() {
        let config = ShellConfig::default();
        let registry = shell_toolkit(config);
        let shell = registry.get("run_shell_command").unwrap();

        let result = shell
            .call(json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert!(result["stdout"].as_str().unwrap().contains("hello"));
        assert_eq!(result["exit_code"], 0);
    }

    #[tokio::test]
    async fn test_blocked_command() {
        let config = ShellConfig::default();
        let registry = shell_toolkit(config);
        let shell = registry.get("run_shell_command").unwrap();

        let result = shell
            .call(json!({"command": "rm -rf /"}))
            .await
            .unwrap();
        assert!(result["error"].as_str().unwrap().contains("blocked"));
    }
}
