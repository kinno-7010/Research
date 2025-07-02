"""
SDO_Mk4_SOHO 太陽物理学研究用解析パッケージ

このパッケージは以下のモジュールから構成されています：

- config: 共通設定、定数、インポート
- sdo_aia_functions: SDO/AIA関連の画像処理・動画作成
- mk4_functions: MK4コロナグラフデータ処理
- lasco_functions: SOHO/LASCO-C2データ処理
- integrated_analysis: 多機器統合解析
- cme_measurement: CME計測・運動学的解析
- claude_analysis_utils: Claude用CME解析ユーティリティ

使用例:
    from SDO_Mk4_SOHO.py_folder import sdo_aia_functions
    from SDO_Mk4_SOHO.py_folder.integrated_analysis import create_single_integrated_image
    from SDO_Mk4_SOHO.py_folder.claude_analysis_utils import run_cme_analysis_workflow
"""

__version__ = "1.0.1"
__author__ = "太陽物理学研究グループ"

# モジュールのインポート
from . import config
from . import sdo_aia_functions
from . import mk4_functions
from . import lasco_functions
from . import integrated_analysis
from . import cme_measurement
from . import claude_analysis_utils

__all__ = [
    'config',
    'sdo_aia_functions', 
    'mk4_functions',
    'lasco_functions',
    'integrated_analysis',
    'cme_measurement',
    'claude_analysis_utils'
]