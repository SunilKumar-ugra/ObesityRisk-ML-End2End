from obesity import logger
from obesity.config import ConfigurationManager
from obesity.components.model_evaluation import ModelEvaluation
from dotenv import load_dotenv

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(eval_config)
        evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        load_dotenv()
        STAGE_NAME = "Evaluation stage"
        logger.info(f"*******************")
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
