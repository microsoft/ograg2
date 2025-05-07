from utils import create_service_context, get_config, load_llm_and_embeds
from qna.tester import AspectTester, RAGASTester
import os

if __name__ == '__main__':
    config = get_config()
    llm, embeddings = load_llm_and_embeds(config.model, config.embedding_model)
    
    answer_dir = os.path.dirname(config.query.answers_file)
    if 'hyperparams' in config.query:
        answer_file = os.path.basename(config.query.answers_file)
        answer_fname, answer_ftype = answer_file.split('.')
        config.query.answers_file = f'{answer_dir}/{answer_fname}_{'_'.join([f'{k}{v}' for k, v in config.query.hyperparams.items()])}.{answer_ftype}' 
        config.evaluator.eval_file = config.query.answers_file

    if 'GuidelineFollow' in config.evaluator.metrics:
        from qna.tester import GuidelineTester
        tester = GuidelineTester(llm=llm)
        if '.json' in config.evaluator.eval_file:
            tester.evaluate_answers_from_json(config.evaluator.eval_file, to_file=True)
        elif '.csv' in config.evaluator.eval_file:
            scores = tester.evaluate_answers_from_csv(config.evaluator.eval_file)

    tester = AspectTester(
                llm=llm,
                metrics=list(filter(lambda x: x != 'GuidelineFollow', config.evaluator.metrics)),
                reference_free=config.evaluator.reference_free,
                type=config.evaluator.type
            )
    if tester.user_msgs != {} and ((not os.path.exists(config.evaluator.eval_file.replace('.json', '_eval_aspects.json'))) or config.rewrite):
        tester.evaluate_answers_from_json(config.evaluator.eval_file, to_file=True)


    tester = RAGASTester(
        llm=llm,
        embeddings=embeddings,
        metrics=config.evaluator.metrics,
    )
    if tester.metrics != [] and ((not os.path.exists(config.evaluator.eval_file.replace('.json', '_eval_ragas.csv'))) or config.rewrite):
        tester.evaluate_answers_from_json(config.evaluator.eval_file, to_file=True)