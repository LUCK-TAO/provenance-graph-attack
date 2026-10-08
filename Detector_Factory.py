"""
Detector Factory - 检测器工厂
根据名称动态创建对应的检测器实例
"""

from Detector.Detector_Interface import BaseDetector


def create_detector(detector_name: str, dataset_name: str = 'cadets',
                    device: str = 'cuda:0', top_k: int = 50, **kwargs) -> BaseDetector:
    """
    工厂函数：根据名称创建检测器实例

    Args:
        detector_name: 检测器名称 ('magic', 'flash', 'orthrus', 'provguard', 'stgan')
        dataset_name: 数据集名称
        device: 计算设备
        top_k: Top-K异常节点数量

    Returns:
        BaseDetector: 检测器实例

    Raises:
        ValueError: 不支持的检测器名称
    """
    detector_name = detector_name.lower()

    if detector_name == 'magic':
        from MAGIC_main.DL_detector import DLDetector
        return DLDetector(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            **kwargs
        )

    elif detector_name == 'flash':
        from Detector.Flash import FlashDetector
        return FlashDetector(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            **kwargs
        )

    elif detector_name == 'orthrus':
        from Detector.Orthrus import OrthrusDetector
        return OrthrusDetector(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            **kwargs
        )

    elif detector_name == 'provguard':
        from Detector.ProvGuard import ProvGuardDetector
        return ProvGuardDetector(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            **kwargs
        )

    elif detector_name == 'stgan':
        from Detector.STGAN import STGANDetector
        return STGANDetector(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            **kwargs
        )

    else:
        raise ValueError(f"不支持的检测器: {detector_name}。"
                         f"支持的检测器: magic, flash, orthrus, provguard, stgan")


def get_supported_detectors() -> list:
    """获取支持的检测器列表"""
    return ['magic', 'flash', 'orthrus', 'provguard', 'stgan']
