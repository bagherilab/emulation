import unittest
from unittest.mock import patch

import logging
import os
from permutation.logger import ExperimentLogger

class TestExperimentLogger(unittest.TestCase):
    def setUp(self):
        self.reset_logging()
        self.experiment_name = "test_experiment"
        self.log_dir = "test_logs"
        self.log_path = f"{self.log_dir}/{self.experiment_name}.log"

        self.logger = ExperimentLogger(
            self.experiment_name, 
            self.log_dir,
        )
    
    def tearDown(self) -> None:
        os.remove(self.log_path)
        os.rmdir(self.log_dir)

    def test_log_with_valid_directory(self):
        message = "Test message"
        self.logger.log(message)

        self.assertTrue(os.path.isfile(self.log_path))
        
        with open(self.log_path, "r") as f:
            log_content = f.read()
            self.assertIn(message, log_content)


    def test_set_up(self):
        format_str = "%(asctime)s:%(name)s:%(message)s"
        level = logging.INFO

        self.assertEqual(self.logger.logger.level, level)

        self.assertEqual(len(self.logger.logger.handlers), 1)
        handler = self.logger.logger.handlers[0]
        self.assertIsInstance(handler, logging.FileHandler)
        baseFilename_chunks = handler.baseFilename.split("/")
        self.assertEqual("/".join(baseFilename_chunks[-2:]), self.logger.log_path)
        self.assertEqual(handler.mode, "w")

        formatter = handler.formatter
        self.assertIsInstance(formatter, logging.Formatter)
        self.assertEqual(formatter._fmt, format_str)

    def reset_logging(self):
        manager = logging.root.manager
        manager.disabled = logging.NOTSET
        for logger in manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.setLevel(logging.NOTSET)
                logger.propagate = True
                logger.disabled = False
                logger.filters.clear()
                handlers = logger.handlers.copy()
                for handler in handlers:
                    # Copied from `logging.shutdown`.
                    try:
                        handler.acquire()
                        handler.flush()
                        handler.close()
                    except (OSError, ValueError):
                        pass
                    finally:
                        handler.release()
                    logger.removeHandler(handler)