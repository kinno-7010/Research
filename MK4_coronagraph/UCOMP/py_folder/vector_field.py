import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import argparse
from pathlib import Path
import json
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict
import warnings

@dataclass
class VectorFieldConfig:
    """ベクトル場設定の管理クラス"""
    # 基本パラメータ
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 1.0
    
    # グリッドパラメータ
    n_radial: int = 10
    n_angular: int = 16
    radial_range: Tuple[float, float] = (0.5, 2.0)
    
    # 可視化パラメータ
    arrow_scale: float = 0.1
    arrow_width: float = 0.003
    arrow_color: str = 'blue'
    arrow_alpha: float = 0.8
    
    # 円の表示設定
    show_circle: bool = True
    circle_color: str = 'red'
    circle_linewidth: float = 2.0
    circle_alpha: float = 0.7
    
    # 図の設定
    figsize: Tuple[float, float] = (10, 10)
    dpi: int = 150
    grid_alpha: float = 0.3
    
    def save_config(self, filepath: Path) -> None:
        """設定をJSONファイルに保存"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: Path) -> 'VectorFieldConfig':
        """JSONファイルから設定を読み込み"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

class CircleNormalVectorField:
    """円の法線方向を基準とするベクトル場可視化クラス"""
    
    def __init__(self, config: VectorFieldConfig):
        self.config = config
        self.fig = None
        self.ax = None
        
    def _create_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """極座標グリッドの生成"""
        r_min, r_max = self.config.radial_range
        r_values = np.linspace(r_min * self.config.radius, 
                              r_max * self.config.radius, 
                              self.config.n_radial)
        theta_values = np.linspace(0, 2 * np.pi, 
                                  self.config.n_angular, 
                                  endpoint=False)
        
        R, THETA = np.meshgrid(r_values, theta_values)
        return R, THETA
    
    def _polar_to_cartesian(self, R: np.ndarray, THETA: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """極座標からデカルト座標への変換"""
        X = self.config.center_x + R * np.cos(THETA)
        Y = self.config.center_y + R * np.sin(THETA)
        return X, Y
    
    def _calculate_normal_angle(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """各点での円の法線方向角度を計算（ラジアン）"""
        dx = X - self.config.center_x
        dy = Y - self.config.center_y
        return np.arctan2(dy, dx)
    
    def _vector_function_example(self, X: np.ndarray, Y: np.ndarray, normal_angles: np.ndarray) -> np.ndarray:
        """
        ベクトル場の角度を定義する関数（例：螺旋パターン）
        
        Parameters:
        -----------
        X, Y : np.ndarray
            デカルト座標
        normal_angles : np.ndarray
            法線方向角度
            
        Returns:
        --------
        np.ndarray
            法線からの偏角（ラジアン）
        """
        # 距離による螺旋効果
        r = np.sqrt((X - self.config.center_x)**2 + (Y - self.config.center_y)**2)
        spiral_factor = 2 * np.pi * (r - self.config.radius) / self.config.radius
        
        # 角度位置による変調
        angular_position = np.arctan2(Y - self.config.center_y, X - self.config.center_x)
        angular_modulation = 0.5 * np.sin(3 * angular_position)
        
        return spiral_factor + angular_modulation
    
    def calculate_vector_components(self, 
                                  vector_angle_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ベクトル成分の計算
        
        Parameters:
        -----------
        vector_angle_func : Callable, optional
            ベクトル角度を計算する関数
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X, Y, U, V座標
        """
        R, THETA = self._create_grid()
        X, Y = self._polar_to_cartesian(R, THETA)
        
        # 法線方向角度の計算
        normal_angles = self._calculate_normal_angle(X, Y)
        
        # ベクトル角度の計算
        if vector_angle_func is None:
            vector_angle_func = self._vector_function_example
        
        deviation_angles = vector_angle_func(X, Y, normal_angles)
        vector_angles = normal_angles + deviation_angles
        
        # ベクトル成分の計算
        U = np.cos(vector_angles)
        V = np.sin(vector_angles)
        
        return X, Y, U, V
    
    def setup_plot(self) -> None:
        """プロット設定の初期化"""
        self.fig, self.ax = plt.subplots(figsize=self.config.figsize, 
                                        dpi=self.config.dpi)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=self.config.grid_alpha)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Vector Field with Circle Normal Reference\n(0° = Normal direction, + = Counter-clockwise)')
    
    def plot_reference_circle(self) -> None:
        """基準円の描画"""
        if self.config.show_circle:
            circle = Circle((self.config.center_x, self.config.center_y), 
                          self.config.radius,
                          fill=False, 
                          color=self.config.circle_color,
                          linewidth=self.config.circle_linewidth,
                          alpha=self.config.circle_alpha)
            self.ax.add_patch(circle)
    
    def plot_vector_field(self, X: np.ndarray, Y: np.ndarray, 
                         U: np.ndarray, V: np.ndarray) -> None:
        """ベクトル場の描画"""
        self.ax.quiver(X, Y, U, V,
                      scale=1/self.config.arrow_scale,
                      width=self.config.arrow_width,
                      color=self.config.arrow_color,
                      alpha=self.config.arrow_alpha,
                      angles='xy', scale_units='xy')
    
    def add_angle_annotations(self, X: np.ndarray, Y: np.ndarray, 
                            normal_angles: np.ndarray, 
                            deviation_angles: np.ndarray) -> None:
        """角度情報の注釈追加"""
        # サンプル点での角度表示
        sample_indices = [(0, 0), (2, 4), (5, 8)]
        
        for i, j in sample_indices:
            if i < X.shape[0] and j < X.shape[1]:
                x, y = X[i, j], Y[i, j]
                normal_deg = np.degrees(normal_angles[i, j])
                deviation_deg = np.degrees(deviation_angles[i, j])
                
                annotation_text = f'Normal: {normal_deg:.1f}°\nDeviation: {deviation_deg:.1f}°'
                self.ax.annotate(annotation_text, 
                               xy=(x, y), 
                               xytext=(10, 10), 
                               textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='yellow', 
                                       alpha=0.7),
                               fontsize=8)
    
    def visualize(self, 
                  vector_angle_func: Optional[Callable] = None,
                  show_annotations: bool = False,
                  save_path: Optional[Path] = None) -> None:
        """
        完全な可視化の実行
        
        Parameters:
        -----------
        vector_angle_func : Callable, optional
            カスタムベクトル角度関数
        show_annotations : bool
            角度注釈の表示
        save_path : Path, optional
            保存パス
        """
        # 計算
        X, Y, U, V = self.calculate_vector_components(vector_angle_func)
        
        # プロット設定
        self.setup_plot()
        
        # 描画
        self.plot_reference_circle()
        self.plot_vector_field(X, Y, U, V)
        
        # 注釈
        if show_annotations:
            normal_angles = self._calculate_normal_angle(X, Y)
            if vector_angle_func is None:
                vector_angle_func = self._vector_function_example
            deviation_angles = vector_angle_func(X, Y, normal_angles)
            self.add_angle_annotations(X, Y, normal_angles, deviation_angles)
        
        # 軸範囲の調整
        r_max = self.config.radial_range[1] * self.config.radius
        margin = 0.5
        self.ax.set_xlim(self.config.center_x - r_max - margin, 
                        self.config.center_x + r_max + margin)
        self.ax.set_ylim(self.config.center_y - r_max - margin, 
                        self.config.center_y + r_max + margin)
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()

def create_custom_vector_functions():
    """カスタムベクトル関数の例"""
    
    def radial_flow(X, Y, normal_angles):
        """純粋な放射状流れ（法線方向）"""
        return np.zeros_like(X)
    
    def tangential_flow(X, Y, normal_angles):
        """純粋な接線方向流れ"""
        return np.full_like(X, np.pi/2)
    
    def spiral_outward(X, Y, normal_angles):
        """外向き螺旋"""
        r = np.sqrt((X - 0)**2 + (Y - 0)**2)
        return np.pi/4 * np.tanh(r - 1)
    
    def wave_pattern(X, Y, normal_angles):
        """波状パターン"""
        angular_pos = np.arctan2(Y, X)
        return 0.3 * np.sin(4 * angular_pos) * np.cos(2 * angular_pos)
    
    def cme_magnetic_field(X, Y, normal_angles):
        """CME様磁場パターン"""
        r = np.sqrt(X**2 + Y**2)
        angular_pos = np.arctan2(Y, X)
        
        # 距離による減衰
        distance_factor = np.exp(-(r - 1)**2 / 0.5)
        
        # 螺旋構造
        spiral_component = 0.5 * np.sin(3 * angular_pos + r)
        
        # 径方向からの偏角
        return distance_factor * spiral_component
    
    return {
        'radial': radial_flow,
        'tangential': tangential_flow,
        'spiral': spiral_outward,
        'wave': wave_pattern,
        'cme_magnetic': cme_magnetic_field
    }

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Circle Normal Vector Field Visualization')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--pattern', type=str, default='spiral',
                       choices=['radial', 'tangential', 'spiral', 'wave', 'cme_magnetic'],
                       help='Vector field pattern')
    parser.add_argument('--annotations', action='store_true', help='Show angle annotations')
    parser.add_argument('--save', type=str, help='Save path for the figure')
    parser.add_argument('--output-dir', type=str, default='/mnt/d/wsl/home/kinno-7010/Research/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # 設定の読み込み
    if args.config:
        config = VectorFieldConfig.load_config(Path(args.config))
    else:
        config = VectorFieldConfig()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベクトル場クラスの初期化
    vector_field = CircleNormalVectorField(config)
    
    # カスタム関数の取得
    vector_functions = create_custom_vector_functions()
    selected_function = vector_functions.get(args.pattern)
    
    # 保存パスの設定
    save_path = None
    if args.save:
        save_path = output_dir / args.save
    elif args.pattern:
        save_path = output_dir / f'vector_field_{args.pattern}.png'
    
    # 可視化の実行
    vector_field.visualize(
        vector_angle_func=selected_function,
        show_annotations=args.annotations,
        save_path=save_path
    )
    
    # 設定の保存
    config_save_path = output_dir / 'vector_field_config.json'
    config.save_config(config_save_path)
    print(f"Configuration saved to: {config_save_path}")

# 使用例とテスト
if __name__ == "__main__":
    # インタラクティブ実行時の例
    if len(sys.argv) == 1:  # スクリプトが直接実行された場合
        # デフォルト設定でのデモ実行
        config = VectorFieldConfig(
            n_radial=8,
            n_angular=12,
            arrow_scale=0.08,
            figsize=(12, 10)
        )
        
        vector_field = CircleNormalVectorField(config)
        
        # CME磁場パターンの可視化
        cme_functions = create_custom_vector_functions()
        vector_field.visualize(
            vector_angle_func=cme_functions['cme_magnetic'],
            show_annotations=True
        )
    else:
        main()