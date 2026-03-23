# -*- coding: utf-8 -*-

import io
import logging
import os
import tempfile
import unittest

from skyllh.core.logging import setup_logging


class SetupLoggingTestCase(unittest.TestCase):
    def setUp(self):
        # Configure a base log_format for testing
        self.cfg = {
            "logging": {
                "log_level": "INFO",
                "log_format": "%(levelname)s:%(name)s:%(message)s",
            },
            "project": {"working_directory": "."},
        }
        self.user_logger_name = "skyllh.tests.setup_logging"

        self._reset_logger("skyllh")
        self._reset_logger(self.user_logger_name)

    def tearDown(self):
        # Needed for safety because logging sets global state
        self._reset_logger("skyllh")
        self._reset_logger(self.user_logger_name)

    def _reset_logger(self, name):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    def _flush_handlers(self, logger_name):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def test_console_handlers(self):
        stream = io.StringIO()

        logger = setup_logging(
            cfg=self.cfg,
            name=self.user_logger_name,
            log_level="INFO",
            console=True,
            reconfigure=True,
        )

        self.assertEqual(logger.name, self.user_logger_name)
        self.assertEqual(logging.getLogger("skyllh").level, logging.INFO)
        self.assertEqual(logging.getLogger(self.user_logger_name).level, logging.INFO)

        # Redirect current console stderr stream to test stream.
        for logger_name in ("skyllh", self.user_logger_name):
            lg = logging.getLogger(logger_name)
            for handler in lg.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.stream = stream

        logger.info("hello")
        self.assertIn("INFO:skyllh.tests.setup_logging:hello", stream.getvalue())

    def test_log_level_from_config(self):
        self.cfg["logging"]["log_level"] = "WARNING"
        stream = io.StringIO()

        logger = setup_logging(
            cfg=self.cfg,
            name=self.user_logger_name,
            reconfigure=True,
        )

        self.assertEqual(logging.getLogger("skyllh").level, logging.WARNING)
        self.assertEqual(
            logging.getLogger(self.user_logger_name).level,
            logging.WARNING,
        )

        for logger_name in ("skyllh", self.user_logger_name):
            lg = logging.getLogger(logger_name)
            for handler in lg.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.stream = stream

        logger.info("should_not_appear")
        logger.warning("should_appear")
        output = stream.getvalue()
        self.assertNotIn("should_not_appear", output)
        self.assertIn("WARNING:skyllh.tests.setup_logging:should_appear", output)

    def test_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.cfg["project"]["working_directory"] = tmpdir
            log_file = "debug.log"

            logger = setup_logging(
                cfg=self.cfg,
                name=self.user_logger_name,
                log_level="DEBUG",
                log_file=log_file,
                reconfigure=True,
            )

            logger.debug("this is a debug message")
            self._flush_handlers(self.user_logger_name)

            log_path = os.path.join(tmpdir, log_file)
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn(
                "DEBUG:skyllh.tests.setup_logging:this is a debug message",
                content,
            )

    def test_reconfigure_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.cfg["project"]["working_directory"] = tmpdir
            log_file = "debug.log"
            log_path = os.path.join(tmpdir, log_file)

            logger = setup_logging(
                cfg=self.cfg,
                name=self.user_logger_name,
                log_file=log_file,
                reconfigure=True,
            )
            logger.info("first")
            self._flush_handlers(self.user_logger_name)
            self.assertTrue(os.path.exists(log_path))

            os.remove(log_path)
            self.assertFalse(os.path.exists(log_path))

            # Reconfigure to drop stale handlers and create fresh ones.
            logger = setup_logging(
                cfg=self.cfg,
                name=self.user_logger_name,
                log_file=log_file,
                reconfigure=True,
            )
            logger.info("second")
            self._flush_handlers(self.user_logger_name)

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("INFO:skyllh.tests.setup_logging:second", content)


if __name__ == "__main__":
    unittest.main()