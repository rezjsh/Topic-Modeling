import torch
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from topic_modeling.pipeline.stage_02_data_transformation import DataTransformationPipeline
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
        data_transformation = data_transformation_pipeline.run_pipeline(df = raw_df.copy(), splits = splits)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

       

    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline execution: {e}")
        raise e


if __name__ == "__main__":
    main()