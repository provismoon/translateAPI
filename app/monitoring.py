import torch
import psutil
import gc

class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process()
        self.reset_peak()
        
    def reset_peak(self):
        self.peak_gpu = 0
        self.peak_ram = 0
        torch.cuda.reset_peak_memory_stats()
    
    def get_memory(self):
        # VRAM (MB)
        gpu_allocated = torch.cuda.memory_allocated() / 1024**2
        gpu_reserved = torch.cuda.memory_reserved() / 1024**2
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        # RAM (MB)
        ram_used = self.process.memory_info().rss / 1024**2
        total_ram = psutil.virtual_memory().total / 1024**2
        ram_used_sys = psutil.virtual_memory().used / 1024**2
        
        self.peak_gpu = max(self.peak_gpu, gpu_allocated)
        self.peak_ram = max(self.peak_ram, ram_used)
        
        return {
            'gpu': {
                'current': gpu_allocated,
                'reserved': gpu_reserved,
                'peak': gpu_peak,
                'peak_tracked': self.peak_gpu
            },
            'ram': {
                'current': ram_used,
                'total': total_ram,
                'peak': self.peak_ram,
                'sys' : ram_used_sys
            }
        }
    
    def log_memory(self, tag=""):
        mem = self.get_memory()
        print(f"\n=== Memory Usage {tag} ===")
        print(f"GPU Memory: {mem['gpu']['current']:.1f}MB (Peak: {mem['gpu']['peak']:.1f}MB)")
        print(f"RAM Memory: {mem['ram']['current']:.1f}MB (Peak: {mem['ram']['peak']:.1f}MB)")
        print(f"RAM Memory sys: {mem['ram']['sys']:.1f}MB (total: {mem['ram']['total']:.1f}MB)")
        
    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()