import typer
import dotenv
from tqdm import tqdm
import yaml

dotenv.load_dotenv()
from stitch.stitch import eval_on_dataset
from stitch.templates import XML_Template, Text_Template, Markdown_Template
from stitch.pydantic_objects import Model, EvalSet
from stitch.data_utils import load_evalset, get_corpus_mapping

SUPPORTED_DOC_MODES = [
        "relevant_first", "random", "relevant_last", "no_context", "only_relevant",
]
SUPPORTED_QUESTION_MODES = ["before", "after", "repeated",  "special"]


def main(
    config: str = typer.Option(
        "default_run.yaml", help="Path to the YAML configuration file."
    ),
    export: bool = typer.Option(
        True, help="Enable exporting results to a file."
    ),
    results_root: str = typer.Option(
        "./results", help="Path to the root directory for the results."
    ),
    print_rolling: bool = typer.Option(
        False, help="Enable rolling accuracy printouts during evaluation."
    )
):
    # Load configuration from YAML file
    print(f"Reading config file: {config}")
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)

    eval_datasets = []
    for dataset_config in tqdm(config_data['eval_sets'], desc="Loading datasets"):
        hf_config = dataset_config['huggingface']
        dataset = load_evalset(hf_config['questions']['repo'], hf_config['questions']['subset'], hf_config['questions']['split'])
        corpus_mapping = None
        if hf_config['requires_corpus']:
            corpus_mapping = get_corpus_mapping(hf_config['corpus']['repo'], hf_config['corpus']['subset'], hf_config['corpus']['split'])

        eval_set = EvalSet(
            dataset=dataset,
            name=dataset_config['name'],
            corpus_mapping=corpus_mapping,
            lang=dataset_config['lang'],
            noise_format=dataset_config['noise_format'],
            noise_key=dataset_config['noise_key'],
            noise_sample=dataset_config['noise_sample'],
            relevant_format=dataset_config['relevant_format'],
            relevant_key=dataset_config['relevant_key'],
            valid_doc_modes=dataset_config['valid_doc_modes']
        )
        eval_datasets.append(eval_set)

    print(f"Using {len(eval_datasets)} datasets.")
    print([x.name for x in eval_datasets])

    models = []
    for model_config in tqdm(config_data['models'], desc="Loading models"):
        model = Model(
            name=model_config['name'],
            provider=model_config['provider'],
            max_context=model_config['max_context'],
            has_special_mode=model_config.get('has_special_mode', False)
        )
        models.append(model)

    print(f"Using {len(models)} models.")
    print([x.name for x in models])

    question_modes = []
    for mode in config_data['question_modes']:
        if mode['name'] in SUPPORTED_QUESTION_MODES:
            question_modes.append(mode['name'])
        else:
            print(f"Warning: {mode['name']} is not a supported question mode.")
    print(f"Using {len(question_modes)} question modes.")
    print(question_modes)


    doc_formatting_options = []
    for mode in config_data['doc_modes']:
        if mode['name'] in SUPPORTED_DOC_MODES:
            doc_formatting_options.append(mode['name'])
        else:
            print(f"Warning: {mode['name']} is not a supported document formatting option.")

    print(f"Using {len(doc_formatting_options)} doc formatting options.")
    print(doc_formatting_options)

    template_config = config_data['templates']
    templates = []
    for template in template_config['defaults']:
        match template['name']:
            case "xml":
                templates.append(XML_Template)
            case "text":
                templates.append(Text_Template)
            case "markdown":
                templates.append(Markdown_Template)
            case other:
                print(f"Warning: {other} is not a supported default template.")

    print(f"Using {len(templates)} templates.")
    print(templates)

    for model in tqdm(models, desc="Models"):
        for evalset in tqdm(eval_datasets, desc="Datasets"):
            for q_mode in tqdm(question_modes, desc="Question modes"):
                for doc_format in tqdm(doc_formatting_options, desc="Doc formats"):
                    if "proxima" in evalset.name:
                        # Hardcoded for now
                        if doc_format == "no_context":
                            print("INFO: skipping no context evaluation for Proxima.")
                            continue
                        if doc_format not in evalset.valid_doc_modes:
                            print(f"INFO: Dataset {evalset.name} does not support {doc_format} mode, skipping...")
                            continue
                        if doc_format == "special" and not model.has_special_mode:
                            print(f"INFO: Model {model.name} does not support special mode, skipping...")
                            continue
                    for template in tqdm(templates, desc="Templates"):
                        print(f"Running {model.name} on {evalset.name} with template {template.name}, {doc_format} and {q_mode}...", flush=True)
                        eval_on_dataset(
                            model=model,
                            evalset=evalset,
                            doc_formatting=doc_format,
                            question_mode=q_mode,
                            template=template,
                            export=export,
                            export_path=results_root,
                            print_rolling=print_rolling,
                        )


if __name__ == "__main__":
    typer.run(main)
