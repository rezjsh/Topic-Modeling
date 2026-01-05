import torch
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from topic_modeling.pipeline.stage_02_data_transformation import DataTransformationPipeline
from topic_modeling.pipeline.stage_03_data_EDA import TopicEDAPipeline
from topic_modeling.pipeline.stage_04_dataset import DatasetPipeline
from topic_modeling.pipeline.stage_05_data_loading import DataLoadingPipeline
from topic_modeling.pipeline.stage_06_model_factory import ModelFactoryPipeline
from topic_modeling.pipeline.stage_07_callbacks import CallbacksPipeline
from topic_modeling.pipeline.stage_08_model_trainer import ModelTrainerPipeline
from topic_modeling.pipeline.stage_09_model_evaluation import ModelEvaluationPipeline
from topic_modeling.pipeline.stage_10_inference import InferencePipeline
from topic_modeling.utils.logging_setup import logger

# Clear CUDA memory (good practice)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logger.info("CUDA cache cleared.")

def main():
    """
    Main execution function to orchestrate the MLOps pipeline stages.
    """
    try:
        config_manager = ConfigurationManager()

        # --- Stage 1: Data Preprocessing (Splitting & Tokenization) ---
        STAGE_NAME = "Stage 01: Data Ingestion"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline(config_manager)
        raw_df, splits = data_ingestion_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 2: Data Transformation (Text Cleaning & Vectorization) ---
        STAGE_NAME = "Stage 02: Data Transformation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation_pipeline = DataTransformationPipeline(config_manager)
        transformation_output = data_transformation_pipeline.run_pipeline(df = raw_df.copy(), splits = splits)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # --- Stage 3: Exploratory Data Analysis (EDA) ---
        STAGE_NAME = "Stage 03: Data EDA"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # Combine raw labels with cleaned text for meaningful EDA
        eda_df = raw_df.copy()
        eda_df['clean_text'] = transformation_output['clean_text']
        topic_eda_pipeline = TopicEDAPipeline(config_manager)
        topic_eda_pipeline.run_pipeline(df = eda_df)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # --- Stage 4: Dataset Creation ---
        STAGE_NAME = "Stage 04: Dataset Creation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        dataset_pipeline = DatasetPipeline(config_manager)
        dataset = dataset_pipeline.run_pipeline(transformation_output=transformation_output, splits=splits)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 5: Data Loading (DataLoader Creation) ---
        STAGE_NAME = "Stage 05: Data Loading"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # TODO: implement data loading pipeline
        data_loading_pipeline = DataLoadingPipeline(config_manager)
        train_loader, val_loader, test_loader = data_loading_pipeline.run_pipeline(dataset=dataset)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # --- Stage 6: Model Factory (Model Initialization) ---
        STAGE_NAME = "Stage 06: Model Factory"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # TODO: implement model factory pipeline
        model_factory_pipeline = ModelFactoryPipeline(config_manager)
        model = model_factory_pipeline.run_pipeline(vocab=transformation_output['vocab'])
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 7: Callbacks Setup ---
        STAGE_NAME = "Stage 07: Callbacks Setup"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # TODO: implement callbacks setup pipeline
        callbacks_pipeline = CallbacksPipeline(config_manager)
        callbacks = callbacks_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # --- Stage 8: Model Trainer (Training Execution) ---
        STAGE_NAME = "Stage 08: Model Trainer"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer_pipeline = ModelTrainerPipeline(config=config_manager, model=model, callbacks=callbacks)
        trained_model = model_trainer_pipeline.run_pipeline(
            transformation_output=transformation_output,
            train_loader=train_loader,
            val_loader=val_loader
        )
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # --- Stage 9: Model Evaluation ---
        STAGE_NAME = "Stage 09: Model Evaluation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # TODO: implement model evaluation pipeline
        model_evaluation_pipeline = ModelEvaluationPipeline(config_manager)
        metrics = model_evaluation_pipeline.run_pipeline(trained_model=trained_model, vocab=transformation_output['vocab'], test_clean_texts=transformation_output['test_clean_text'])
        logger.info(f"Evaluation Results: {metrics}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # --- Stage 10: Model Inference ---
        STAGE_NAME = "Stage 10: Model Inference"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # TODO: implement model inference pipeline
        model_inference_pipeline = InferencePipeline(config_manager)
        # Unseen User Data
        unseen_data = [
            "The latest space mission discovered water on Mars.",
            "The stock market crashed after the Federal Reserve interest rate hike.",
            "New research in deep learning suggests transformers are efficient."
        ]
        results = model_inference_pipeline.run_pipeline(unseen_data)
        # Display
        for i, res in enumerate(results):
            logger.info(f"\nText: {unseen_data[i][:50]}...")
            logger.info(f"Detected Topic {res['topic_id']} (Confidence: {res['confidence']:.2%})")
            logger.info(f"Keywords: {', '.join(res['keywords'])}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline execution: {e}")
        raise e


if __name__ == "__main__":
    main()