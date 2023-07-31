from YoloBodyDetectionToCleanedCsv.YoloBodyDetectionToCsv.YoloBodyDetectionToCsv import YoloBodyDetectionToCsv
from YoloBodyDetectionToCleanedCsv.YoloBodyCsvToCleanedCsv.YoloBodyCsvToCleanedCsv import YoloBodyCsvToCleanedCsv
#
def YoloBodyDetectionToCleanedCsv(PathSubFolderToAnalyze):
    PathSubFolderToAnalyze = PathSubFolderToAnalyze
    YoloBodyDetectionToCsv(PathSubFolderToAnalyze)
    YoloBodyCsvToCleanedCsv(PathSubFolderToAnalyze)
    return 0