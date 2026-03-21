# -*- coding: utf-8 -*-

import io
import logging
import os
import tempfile
import unittest

from skyllh.core.debugging import configure_logging


class ConfigureLogging_TestCase(
        unittest.TestCase,
):
    def setUp(self):
        self.cfg = {
            'debugging': {'log_format': '%(levelname)s:%(name)s:%(message)s'},
            'project': {'working_directory': '.'},
        }
        self.script_logger_name = 'skyllh.tests.configure_logging'
        self._reset_logger('skyllh')
        self._reset_logger(self.script_logger_name)

    def tearDown(self):
        self._reset_logger('skyllh')
        self._reset_logger(self.script_logger_name)

    def _reset_logger(self, name):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)

    def test_configure_logging_console_handlers(self):
        stream = io.StringIO()

        script_logger = configure_logging(
            cfg=self.cfg,
            script_logger_name=self.script_logger_name,
            log_level=logging.INFO
        )

        self.assertEqual(script_logger.name, self.script_logger_name)
        self.assertEqual(script_logger.level, logging.INFO)
        self.assertEqual(logging.getLogger('skyllh').level, logging.INFO)

        logger = logging.getLogger(self.script_logger_name)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = stream

        script_logger.info('hello')
        self.assertIn('INFO:skyllh.tests.configure_logging:hello', stream.getvalue())

    def test_configure_logging_debug_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.cfg['project']['working_directory'] = tmpdir
            filename = 'debug.log'

            script_logger = configure_logging(
                cfg=self.cfg,
                script_logger_name=self.script_logger_name,
                log_level=logging.DEBUG,
                debug_pathfilename=filename
            )

            script_logger.debug('this is a debug message')
            for handler in logging.getLogger(self.script_logger_name).handlers:
                handler.flush()

            log_path = os.path.join(tmpdir, filename)
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            self.assertIn('DEBUG:skyllh.tests.configure_logging:this is a debug message', log_content)


if __name__ == '__main__':
    unittest.main()
