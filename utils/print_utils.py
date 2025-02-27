from rich.console import Console
from rich.table import Table
from rich.panel import Panel
console = Console()

def print_config(self) -> None:
    console.print(Panel.fit(str(self.config),
                        title="[bold green]√ Experiment Configuration [/]",
                        border_style="green"))
    
def print_result_table(self, i: int, mean_time_generate: float, eer: float, threshold: float, 
                    DI:float,mean_time_genuine: float =0.0, mean_time_impostor: float = 0.0) -> None:
    """打印实验结果表格
    
    Args:
        self (Experiment): 实验对象
        i (int): 实验次数
        mean_time_generate (float): 平均生成时间（秒）
        eer (float): 等错误率
        threshold (float): 最佳阈值
        DI (float): Decidability Index
        mean_time_genuine (float, optional): 真匹配平均时间（秒）
        mean_time_impostor (float, optional): 假匹配平均时间（秒）
    """
    method_name = self.config.method_config.method_name
    dataset_name = self.config.dataset_config.dataset_name
    # Create result table
    table = Table(title=f"{method_name} on {dataset_name} \n Experiment ({i}/{self.config.expriment_times}) result, ", 
            show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Mean Generation Time", f"{mean_time_generate*1000:.3f}ms")  # mean_time is in seconds
    
    # Add matching times if provided
    if mean_time_genuine is not None:
        table.add_row("Mean Genuine Match Time", f"{mean_time_genuine*1000:.3f}ms")
    if mean_time_impostor is not None:
        table.add_row("Mean Impostor Match Time", f"{mean_time_impostor*1000:.3f}ms")
        
    table.add_row("Equal Error Rate (EER)", f"{eer*100:.2f}%")  # EER is already percentage
    table.add_row("Optimal Threshold", f"{threshold:.4f}")
    table.add_row("Decidability Index (DI)", f"{DI:.4f}")
    console.print(table)