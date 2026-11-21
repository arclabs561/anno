use rusqlite::{ffi, Connection, Error};
use std::{env, path::Path, process};

fn main() {
    let args: Vec<_> = env::args().collect();
    let [_, action, path, version] = args.as_slice() else {
        panic!("usage: probe seed|reopen|verify DATABASE SQLITE_VERSION");
    };
    assert_eq!(rusqlite::version(), version);
    let conn = Connection::open(Path::new(path)).unwrap();
    match action.as_str() {
        "seed" => {
            let mode: String = conn
                .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
                .unwrap();
            assert_eq!(mode, "wal");
            conn.execute_batch(
                "PRAGMA wal_autocheckpoint=0;
                 CREATE TABLE scores (id INTEGER PRIMARY KEY, f1 REAL);
                 INSERT INTO scores VALUES (1, 0.90);",
            )
            .unwrap();
            // Simulate a writer exiting after COMMIT but before closing SQLite.
            process::exit(0);
        }
        "reopen" => {
            let first = conn.query_row("SELECT f1 FROM scores WHERE id=1", [], |r| {
                r.get::<_, f64>(0)
            });
            if let Err(Error::SqliteFailure(e, _)) = &first {
                if e.extended_code == ffi::SQLITE_IOERR_TRUNCATE {
                    println!("SQLITE_IOERR_TRUNCATE");
                    process::exit(42);
                }
            }
            assert_eq!(first.unwrap(), 0.90);
            conn.execute("INSERT INTO scores VALUES (2, 0.30)", [])
                .unwrap();
        }
        "verify" => {
            let mut stmt = conn.prepare("SELECT f1 FROM scores ORDER BY id").unwrap();
            let scores = stmt
                .query_map([], |r| r.get::<_, f64>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            assert_eq!(scores, [0.90, 0.30]);
            let integrity: String = conn
                .query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .unwrap();
            assert_eq!(integrity, "ok");
        }
        _ => panic!("unknown action {action}"),
    }
}
