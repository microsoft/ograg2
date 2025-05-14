import os
from utils import get_documents, load_llm_and_embeds, get_config
import pickle as pkl
from llama_index.core.schema import Document

MAX_TOKENS = 4096

RULE_QUESTION_TMPL = """
Given the following data and a set of deductive rules, generate a hard question that require the application of the rules on the data to generate the answer. 

Data: {data}

Rules: {rules}

Question:
"""

RULE_ANSWER_TMPL = """
Given the following data and a set of deductive rules, generate the answer to following question while applying of the rules on the data to generate the answer. 

Data: {data}

Rules: {rules}

Question: {question}

Answer: 
"""

CHECK_QUESTION_TMPL = """
Given the following data and a set of deductive rules, check if the given question requires the application of the rules on the data to generate the given answer. Rate the question from 1-10, where 10 denotes a very good application of the rules and 1 denotes a bad application of the rules.
Print the reasoning as "Reasoning: <>" followed by the rating as "Rating: <rating>", where <rating> is an integer from 1-10.

Data: {data}

Rules: {rules}

Question: {question}

Answer: {answer}

Reasoning: 
Rating:
"""



if __name__ == '__main__':
    config = get_config()
    llm, embed_llm = load_llm_and_embeds(config.model, config.embedding_model) 

    documents = get_documents(config.data.documents_dir, subdir=config.data.subdir, smart_pdf=config.data.smart_pdf, full_text=config.data.full_text)

    if config.question_generator.framework == 'ragas':
        from ragas.testset.generator import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context

        generator = TestsetGenerator.from_langchain(
            generator_llm=llm, critic_llm=llm, embeddings=embed_llm, chunk_size=8192
        )
        emb_dir = config.embedding_model.deployment_name

        # if os.path.exists(f"{config.data.documents_dir}/{emb_dir}_docstore.pkl"):
        #     generator.docstore = pkl.load(open(f"{config.data.documents_dir}/{emb_dir}_docstore.pkl", 'rb'))
        #     testset = generator.generate_with_docstore(
        #         test_size=config.question_generator.test_size,
        #         raise_exceptions=False,
        #         with_debugging_logs=False,
        #         distributions={
        #             simple: config.question_generator.distr.simple, 
        #             reasoning: config.question_generator.distr.reasoning, 
        #             multi_context: config.question_generator.distr.multi_context
        #         },
        #     )
        # else:
        testset = generator.generate_with_llamaindex_docs(
            documents=documents,
            test_size=config.question_generator.test_size,
            raise_exceptions=False,
            with_debugging_logs=False,
            distributions={
                simple: config.question_generator.distr.simple, 
                reasoning: config.question_generator.distr.reasoning, 
                multi_context: config.question_generator.distr.multi_context},
        )
        # os.makedirs(os.path.dirname(f"{config.data.documents_dir}/{emb_dir}_docstore.pkl"), exist_ok=True)
        # pkl.dump(generator.docstore, open(f"{config.data.documents_dir}/{emb_dir}_docstore.pkl", 'wb'))

        os.makedirs(f"{config.data.documents_dir}/questions/{config.question_generator.framework}", exist_ok=True)
        distr_str = '_'.join([str(x).replace('.', 'p') for x in config.question_generator.distr.values()])
        output_fname = f"{config.data.documents_dir}/questions/{config.question_generator.framework}/testset{config.question_generator.test_size}_{distr_str}.csv"
        testset.to_pandas().to_csv(output_fname, index=False)
        

    elif config.question_generator.framework == 'ragas_onto':
        from ragas.testset.generator import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context
        from utils import load_graph_nodes
        nodes = load_graph_nodes(ontology_nodes_path=config.data.kg_storage_path)
        documents = [Document(text=str(node)) for node in nodes]

        generator = TestsetGenerator.from_langchain(
            generator_llm=llm, critic_llm=llm, embeddings=embed_llm, chunk_size=8192
        )
        emb_dir = config.embedding_model.deployment_name

        if os.path.exists(f"{config.data.kg_storage_path}/{emb_dir}_docstore.pkl"):
            generator.docstore = pkl.load(open(f"{config.data.kg_storage_path}/{emb_dir}_docstore.pkl", 'rb'))
            testset = generator.generate_with_docstore(
                test_size=config.question_generator.test_size,
                raise_exceptions=False,
                with_debugging_logs=False,
                distributions={
                    simple: config.question_generator.distr.simple, 
                    reasoning: config.question_generator.distr.reasoning, 
                    multi_context: config.question_generator.distr.multi_context
                },
            )
        else:
            testset = generator.generate_with_llamaindex_docs(
                documents=documents,
                test_size=config.question_generator.test_size,
                raise_exceptions=False,
                with_debugging_logs=False,
                distributions={
                    simple: config.question_generator.distr.simple, 
                    reasoning: config.question_generator.distr.reasoning, 
                    multi_context: config.question_generator.distr.multi_context},
            )
            # os.makedirs(os.path.dirname(f"{config.data.kg_storage_path}/{emb_dir}_docstore.pkl"), exist_ok=True)
            # pkl.dump(generator.docstore, open(f"{config.data.kg_storage_path}/{emb_dir}_docstore.pkl", 'wb'))

        os.makedirs(f"{config.data.kg_storage_path}/questions/{config.question_generator.framework}", exist_ok=True)
        distr_str = '_'.join([str(x).replace('.', 'p') for x in config.question_generator.distr.values()])
        output_fname = f"{config.data.kg_storage_path}/questions/{config.question_generator.framework}/testset{config.question_generator.test_size}_{distr_str}.csv"
        testset.to_pandas().to_csv(output_fname, index=False)

    elif config.question_generator.framework == 'ontodoc_ragas':
        from ragas.testset.generator import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context
        from utils import load_graph_nodes
        from qna.ontology_docstore import OntologyDocStore

        nodes = load_graph_nodes(ontology_nodes_path=config.data.kg_storage_path)
        documents = [Document(text=str(node)) for node in nodes]
        generator = TestsetGenerator.from_langchain(
            generator_llm=llm, critic_llm=llm, embeddings=embed_llm, chunk_size=8192
        )
        
        docstore = OntologyDocStore()
        docstore.add_onto_mappings(config.data.kg_storage_path)

        testset = generator.generate_with_docstore(
            docstore=docstore,
            test_size=config.question_generator.test_size,
            raise_exceptions=False,
            with_debugging_logs=True,
            distributions={
                simple: config.question_generator.distr.simple, 
                reasoning: config.question_generator.distr.reasoning, 
                multi_context: config.question_generator.distr.multi_context
            },
        )

        os.makedirs(f"{config.data.kg_storage_path}/questions/{config.question_generator.framework}", exist_ok=True)
        distr_str = '_'.join([str(x).replace('.', 'p') for x in config.question_generator.distr.values()])
        output_fname = f"{config.data.kg_storage_path}/questions/{config.question_generator.framework}/testset{config.question_generator.test_size}_{distr_str}.csv"
        testset.to_pandas().to_csv(output_fname, index=False)

    elif config.question_generator.framework == 'ontodoc_ragas_rules':
        from ragas.testset.generator import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context
        from utils import load_graph_nodes
        from qna.ontology_docstore import OntologyDocStore
        import pandas as pd
        
        rules = []
        if 'rules_file' in config.data and config.data.rules_file:
            with open (config.data.rules_file) as f:
                rules += [line[:-1] for line in f.readlines()]


        nodes = load_graph_nodes(ontology_nodes_path=config.data.kg_storage_path)
        documents = [Document(text=str(node)) for node in nodes]
        generator = TestsetGenerator.from_langchain(
            generator_llm=llm, critic_llm=llm, embeddings=embed_llm, chunk_size=8192
        )
        
        docstore = OntologyDocStore()
        docstore.add_onto_mappings(config.data.kg_storage_path)
        
        testset = {'question': [], 'context': [], 'ground_truth': []}
        rules = '\n'.join(rules)
        threshold = 7
        num_nodes = 5
        nodes = docstore.get_random_nodes(k=num_nodes)
        while len(testset['question']) < config.question_generator.test_size:
            context = '\n'.join(map(lambda x: x.page_content, nodes))
            question = llm.invoke(RULE_QUESTION_TMPL.format(data=context, rules=rules), max_tokens=MAX_TOKENS).content
            answer = llm.invoke(RULE_ANSWER_TMPL.format(data=context, rules=rules, question=question), max_tokens=MAX_TOKENS).content
            rating = llm.invoke(CHECK_QUESTION_TMPL.format(data=context, rules=rules, question=question, answer=answer), max_tokens=MAX_TOKENS).content
            # print (question)
            rating = rating.split("Rating: ")[1]
            try:
                if float(rating) > threshold:
                    testset['question'].append(question)
                    testset['ground_truth'].append(answer)
                    testset['context'].append(context)
                    nodes = [docstore.get_similar(node) for node in nodes]
                    continue
            except:
                pass
            nodes = docstore.get_random_nodes(k=num_nodes)

        os.makedirs(f"{config.data.kg_storage_path}/questions/{config.question_generator.framework}", exist_ok=True)
        distr_str = '_'.join([str(x).replace('.', 'p') for x in config.question_generator.distr.values()])
        output_fname = f"{config.data.kg_storage_path}/questions/{config.question_generator.framework}/testset_rules{config.question_generator.test_size}.csv"
        pd.DataFrame(testset).to_csv(output_fname, index=False)

    elif config.question_generator.framework == 'my_model':
        raise NotImplementedError("My model is not implemented yet.")