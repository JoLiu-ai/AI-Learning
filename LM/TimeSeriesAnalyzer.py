import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

class TimeSeriesAnalyzer:
    def __init__(self, data, lags=40):
        """
        初始化时间序列分析器
        
        参数:
        data: 时间序列数据
        lags: 需要计算的滞后阶数
        """
        self.data = data
        self.lags = lags
        
    def plot_correlation_analysis(self):
        """绘制ACF和PACF图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制ACF
        plot_acf(self.data, lags=self.lags, ax=ax1)
        ax1.set_title('自相关函数(ACF)')
        ax1.grid(True)
        
        # 绘制PACF
        plot_pacf(self.data, lags=self.lags, ax=ax2)
        ax2.set_title('偏自相关函数(PACF)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def detailed_correlation_analysis(self):
        """详细的相关性分析"""
        # 计算ACF和PACF值
        acf_values = acf(self.data, nlags=self.lags)
        pacf_values = pacf(self.data, nlags=self.lags)
        
        # 计算置信区间
        confidence_interval = 1.96 / np.sqrt(len(self.data))
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'Lag': range(len(acf_values)),
            'ACF': acf_values,
            'PACF': pacf_values,
            'CI_upper': confidence_interval,
            'CI_lower': -confidence_interval
        })
        
        # 找出显著的相关性
        significant_acf = results[abs(results['ACF']) > confidence_interval]
        significant_pacf = results[abs(results['PACF']) > confidence_interval]
        
        return results, significant_acf, significant_pacf
    
    def suggest_arima_orders(self):
        """基于ACF和PACF图提供ARIMA模型阶数建议"""
        acf_values = acf(self.data, nlags=self.lags)
        pacf_values = pacf(self.data, nlags=self.lags)
        
        # 计算显著性界限
        significance_level = 1.96 / np.sqrt(len(self.data))
        
        # 分析ACF衰减模式
        acf_significant = np.where(abs(acf_values) > significance_level)[0]
        pacf_significant = np.where(abs(pacf_values) > significance_level)[0]
        
        # 提供建议
        suggestions = {
            'AR_order': len(pacf_significant),
            'MA_order': len(acf_significant),
            'suggested_model': f"ARIMA({len(pacf_significant)},d,{len(acf_significant)})"
        }
        
        return suggestions

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_points = 1000
    
    # 创建AR(1)过程
    ar_params = [0.8]
    ar_data = np.zeros(n_points)
    for t in range(1, n_points):
        ar_data[t] = ar_params[0] * ar_data[t-1] + np.random.normal(0, 1)
    
    # 初始化分析器
    analyzer = TimeSeriesAnalyzer(ar_data)
    
    # 绘制相关图
    analyzer.plot_correlation_analysis()
    
    # 获取详细分析
    results, sig_acf, sig_pacf = analyzer.detailed_correlation_analysis()
    
    # 获取ARIMA建议
    suggestions = analyzer.suggest_arima_orders()
    
    print("\nARIMA模型建议:")
    print(f"建议的模型: {suggestions['suggested_model']}")
    print(f"AR阶数: {suggestions['AR_order']}")
    print(f"MA阶数: {suggestions['MA_order']}")
