use std::process::{Command, Output};

fn anno(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_anno"))
        .args(args)
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap()
}

#[test]
fn gated_names_and_aliases_route_help_to_the_command() {
    for name in [
        "cross-doc",
        "coalesce",
        "benchmark",
        "bench",
        "sampler",
        "muxer",
        "history",
    ] {
        for args in [[name, "--help"], ["help", name]] {
            let output = anno(&args);
            if cfg!(feature = "eval") {
                assert!(output.status.success(), "{args:?}");
                assert!(
                    String::from_utf8_lossy(&output.stdout).contains("Usage:"),
                    "{args:?}"
                );
            } else {
                assert_eq!(output.status.code(), Some(2), "{args:?}");
                assert!(output.stdout.is_empty());
                let diagnostic = String::from_utf8(output.stderr).unwrap();
                assert!(
                    diagnostic.contains(name) && diagnostic.contains("--features eval"),
                    "{diagnostic}"
                );
            }
        }
    }
}

#[cfg(not(feature = "eval"))]
#[test]
fn gated_command_shapes_fail_and_prose_still_extracts() {
    for args in [
        &["cross-doc"][..],
        &["coalesce", "./documents"],
        &["cross-doc", "--import", "saved.json"],
        &["--color", "never", "history", "stats"],
        &["--color=never", "muxer", "decide"],
    ] {
        let output = anno(args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("--features eval"));
    }
    let invalid_color = anno(&["--color", "purple", "history", "stats"]);
    assert_eq!(invalid_color.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&invalid_color.stderr).contains("purple"));
    for args in [
        &["history", "of", "navigation", "in", "2026-11-25"][..],
        &["history stats in 2026-11-25"],
        &["--", "cross-doc", "--help", "2026-11-25"],
    ] {
        let output = anno(args);
        assert!(output.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("2026-11-25"),
            "{args:?}"
        );
    }
}

#[test]
fn regular_help_and_active_commands_still_work() {
    for args in [&["--help"][..], &["help", "extract"], &["x", "--help"]] {
        let output = anno(args);
        assert!(output.status.success(), "{args:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
    }
}
