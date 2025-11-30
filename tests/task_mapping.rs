//! Tests for task-dataset-backend mapping system.

use anno::eval::task_mapping::{
    get_dataset_tasks, get_task_backends, get_task_datasets, Task, TaskMapping,
};

#[test]
fn test_ner_datasets() {
    let datasets = get_task_datasets(Task::NER);
    assert!(!datasets.is_empty(), "NER should have datasets");
    assert!(datasets.contains(&anno::eval::loader::DatasetId::WikiGold));
    assert!(datasets.contains(&anno::eval::loader::DatasetId::CoNLL2003Sample));
}

#[test]
fn test_dataset_tasks() {
    let tasks = get_dataset_tasks(anno::eval::loader::DatasetId::WikiGold);
    assert!(tasks.contains(&Task::NER));
}

#[test]
fn test_gliner2_capabilities() {
    let backends = get_task_backends(Task::NER);
    assert!(backends.contains(&"gliner2"));

    let backends_re = get_task_backends(Task::RelationExtraction);
    assert!(
        backends_re.contains(&"gliner2"),
        "GLiNER2 should support relation extraction"
    );
}

#[test]
fn test_task_mapping_build() {
    let mapping = TaskMapping::build();

    // Check NER mappings
    let ner_datasets = mapping.datasets_for_task("ner");
    assert!(ner_datasets.is_some());
    assert!(!ner_datasets.unwrap().is_empty());

    // Check GLiNER2 capabilities
    let gliner2_tasks = mapping.tasks_for_backend("gliner2");
    assert!(gliner2_tasks.is_some());
    let tasks = gliner2_tasks.unwrap();
    assert!(tasks.contains(&"ner".to_string()));
    assert!(
        tasks.contains(&"re".to_string()),
        "GLiNER2 should support relation extraction"
    );
}

#[test]
fn test_coref_datasets() {
    let datasets = get_task_datasets(Task::IntraDocCoref);
    assert!(!datasets.is_empty());
    assert!(datasets.contains(&anno::eval::loader::DatasetId::GAP));
    assert!(datasets.contains(&anno::eval::loader::DatasetId::PreCo));
}

#[test]
fn test_relation_extraction_datasets() {
    let datasets = get_task_datasets(Task::RelationExtraction);
    assert!(!datasets.is_empty());
    assert!(datasets.contains(&anno::eval::loader::DatasetId::DocRED));
    assert!(datasets.contains(&anno::eval::loader::DatasetId::ReTACRED));
}

#[test]
fn test_w2ner_discontinuous() {
    let backends = get_task_backends(Task::DiscontinuousNER);
    assert!(
        backends.contains(&"w2ner"),
        "W2NER should support discontinuous NER"
    );
}

#[test]
fn test_multi_task_backends() {
    // GLiNER2 should support multiple tasks
    let ner_backends = get_task_backends(Task::NER);
    let re_backends = get_task_backends(Task::RelationExtraction);

    // GLiNER2 should appear in both
    assert!(ner_backends.contains(&"gliner2"));
    assert!(
        re_backends.contains(&"gliner2"),
        "GLiNER2 should support both NER and RE"
    );
}
