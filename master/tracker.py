import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from union_find import UnionFind

class DependencyTracker:
    def __init__(self):
        self.dependencies = defaultdict(list)
        self.hooks = []
        self.layer_info = {}  # Store layer dimension information
        self._setup_op_hooks()

    def _setup_op_hooks(self):
        self._original_cat = torch.cat
        self._original_chunk = torch.chunk
        self._original_add = torch.Tensor.__add__
        
        # Store all activation functions and parameter-free layers that need to be tracked
        self._original_activations = {
            'maxpool': nn.functional.max_pool2d,  # Handle maxpool as a parameter-free layer
            'relu': nn.functional.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'leaky_relu': nn.functional.leaky_relu,
            'elu': nn.functional.elu,
            'selu': nn.functional.selu,
            'gelu': nn.functional.gelu,
            'prelu': nn.functional.prelu,
            'softmax': nn.functional.softmax,
            'mish': nn.functional.mish
        }
        
        def cat_hook(*args, **kwargs):
            tensors = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
            dim = kwargs.pop('dim', 1)  # Default to channel dimension and remove from kwargs
            
            # Create operation identifier
            op_name = f"cat_{len(self.layer_info)}"
            
            # Execute original operation
            result = self._original_cat(tensors, dim=dim, **kwargs)
            
            # Process dependencies for structured pruning
            channel_offset = 0
            for tensor in tensors:
                if hasattr(tensor, '_source_info'):
                    src_layer = tensor._source_info['layer']
                    n_channels = tensor.size(1)
                    
                    # In concatenation, each input channel maps to exactly one output channel
                    # The dependency is one-to-one between source output and concat input/output
                    for i in range(n_channels):
                        src_key = (src_layer, 0, i)  # Source layer output (0 means output)
                        cat_in_key = (op_name, 1, channel_offset + i)  # Cat layer input (1 means input)
                        cat_out_key = (op_name, 0, channel_offset + i)  # Cat layer output
                        
                        # Connect source to concat input
                        self.dependencies[src_key].append(cat_in_key)
                        self.dependencies[cat_in_key].append(src_key)
                        
                        # Connect concat input to output
                        self.dependencies[cat_in_key].append(cat_out_key)
                        self.dependencies[cat_out_key].append(cat_in_key)
                    
                    channel_offset += n_channels
            
            # Add source info to output for tracking
            result._source_info = {
                'layer': op_name,
                'channels': channel_offset
            }
            self.layer_info[op_name] = {'type': 'cat', 'channels': channel_offset}
            
            return result
            
        def chunk_hook(tensor, chunks, dim=0):
            base_name = f"chunk_{len(self.layer_info)}"
            
            # Execute original operation
            results = self._original_chunk(tensor, chunks, dim=dim)
            
            if hasattr(tensor, '_source_info'):
                chunk_size = tensor.size(1) // chunks
                src_layer = tensor._source_info['layer']
                
                # Process each output chunk
                for i, result_tensor in enumerate(results):
                    chunk_name = f"{base_name}_{i}"
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size
                    
                    # In chunk/split operations, maintain precise channel mapping
                    # Each output chunk gets a specific portion of input channels
                    for j in range(start_idx, end_idx):
                        src_key = (src_layer, 0, j)  # Source layer output (0 means output)
                        chunk_in_key = (chunk_name, 1, j - start_idx)  # Chunk input (1 means input)
                        chunk_out_key = (chunk_name, 0, j - start_idx)  # Chunk output
                        
                        # Connect source to chunk input
                        self.dependencies[src_key].append(chunk_in_key)
                        self.dependencies[chunk_in_key].append(src_key)
                        
                        # Connect chunk input to output
                        self.dependencies[chunk_in_key].append(chunk_out_key)
                        self.dependencies[chunk_out_key].append(chunk_in_key)
                    
                    # Add source info to each output
                    result_tensor._source_info = {
                        'layer': chunk_name,
                        'channels': chunk_size
                    }
                    self.layer_info[chunk_name] = {'type': 'chunk', 'channels': chunk_size}
            
            return results
            
        tracker = self  # Save reference to DependencyTracker instance
        
        def add_hook(tensor1, tensor2):
            op_name = f"add_{len(tracker.layer_info)}"
            
            # Execute original operation
            result = tracker._original_add(tensor1, tensor2)
            
            # Process dependencies for structured pruning
            if isinstance(tensor1, torch.Tensor):
                n_channels = tensor1.size(1)
                for tensor in [tensor1, tensor2]:
                    if hasattr(tensor, '_source_info'):
                        src_layer = tensor._source_info['layer']
                        # Element-wise addition requires matching dimensions
                        for i in range(n_channels):
                            src_key = (src_layer, 0, i)  # Source layer output (0 means output)
                            target_key = (op_name, 1, i)  # Addition input (1 means input)
                            tracker.dependencies[src_key].append(target_key)
                            tracker.dependencies[target_key].append(src_key)
                            
                            # In addition, both inputs must have the same dimensions
                            for other_tensor in [tensor1, tensor2]:
                                if id(other_tensor) != id(tensor) and hasattr(other_tensor, '_source_info'):
                                    other_key = (other_tensor._source_info['layer'], 0, i)  # Output of other tensor
                                    # Bidirectional dependency for addition between inputs
                                    tracker.dependencies[src_key].append(other_key)
                                    tracker.dependencies[other_key].append(src_key)
                            
                            # Add dependencies between inputs and addition result
                            add_out_key = (op_name, 0, i)  # Addition layer output
                            tracker.dependencies[src_key].append(add_out_key)
                            tracker.dependencies[add_out_key].append(src_key)
                
                # Add source info to result
                result._source_info = {
                    'layer': op_name,
                    'channels': n_channels
                }
                tracker.layer_info[op_name] = {'type': 'add', 'channels': n_channels}
            
            return result
        
        def activation_hook(name):
            def hook(*args, **kwargs):
                # Get input tensor
                input_tensor = args[0]
                
                # Create operation identifier
                op_name = f"{name}_{len(tracker.layer_info)}"
                
                # Execute original operation
                result = tracker._original_activations[name](*args, **kwargs)
                
                if isinstance(input_tensor, torch.Tensor) and hasattr(input_tensor, '_source_info'):
                    n_channels = input_tensor.size(1)
                    src_layer = input_tensor._source_info['layer']
                    
                    # Process each channel
                    for i in range(n_channels):
                        src_key = (src_layer, 0, i)  # Source layer output
                        act_key = (op_name, 0, i)  # Activation output
                        
                        # Connect source to activation function
                        tracker.dependencies[src_key].append(act_key)
                        tracker.dependencies[act_key].append(src_key)
                    
                    # Add source info to result
                    result._source_info = {
                        'layer': op_name,
                        'channels': n_channels
                    }
                    tracker.layer_info[op_name] = {'type': name, 'channels': n_channels}
                
                return result
            return hook
            
        # Replace original operations with hooked versions
        torch.cat = cat_hook
        torch.chunk = chunk_hook
        torch.Tensor.__add__ = add_hook
        
        # Replace all activation functions
        for act_name, act_fn in self._original_activations.items():
            if act_name == 'relu':
                nn.functional.relu = activation_hook('relu')
            elif act_name == 'sigmoid':
                torch.sigmoid = activation_hook('sigmoid')
            elif act_name == 'tanh':
                torch.tanh = activation_hook('tanh')
            elif act_name == 'leaky_relu':
                nn.functional.leaky_relu = activation_hook('leaky_relu')
            elif act_name == 'elu':
                nn.functional.elu = activation_hook('elu')
            elif act_name == 'selu':
                nn.functional.selu = activation_hook('selu')
            elif act_name == 'gelu':
                nn.functional.gelu = activation_hook('gelu')
            elif act_name == 'prelu':
                nn.functional.prelu = activation_hook('prelu')
            elif act_name == 'softmax':
                nn.functional.softmax = activation_hook('softmax')
            elif act_name == 'mish':
                nn.functional.mish = activation_hook('mish')
            elif act_name == 'maxpool':
                nn.functional.max_pool2d = activation_hook('maxpool')

    def register_hooks(self, model: nn.Module):
        def get_layer_hook(layer_name: str, module: nn.Module):
            def hook(module, inputs, output):
                input_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
                
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Handle Conv2d and Linear layers
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                    
                    self.layer_info[layer_name] = {
                        'type': module.__class__.__name__,
                        'in_channels': in_features,
                        'out_channels': out_features
                    }
                    
                    if hasattr(input_tensor, '_source_info'):
                        src_layer = input_tensor._source_info['layer']
                        for i in range(in_features):
                            src_key = (src_layer, 0, i)
                            for o in range(out_features):
                                out_key = (layer_name, 0, o)
                                in_key = (layer_name, 1, i)
                                
                                self.dependencies[src_key].append(in_key)
                                self.dependencies[in_key].append(src_key)
                    
                    n_channels = out_features

                elif isinstance(module, nn.BatchNorm2d):
                    # Handle BatchNorm layers
                    n_channels = input_tensor.size(1)  # Get number of channels from input
                    
                    self.layer_info[layer_name] = {
                        'type': module.__class__.__name__,
                        'channels': n_channels
                    }
                    
                    # BatchNorm preserves channel structure
                    if hasattr(input_tensor, '_source_info'):
                        src_layer = input_tensor._source_info['layer']
                        for i in range(n_channels):
                            src_key = (src_layer, 0, i)
                            cur_key = (layer_name, 0, i)  # Same input/output channels
                            
                            # Establish dependency between input and output channels
                            self.dependencies[src_key].append(cur_key)
                            self.dependencies[cur_key].append(src_key)
                
                # Add source info to output for tracking
                if isinstance(output, torch.Tensor):
                    output._source_info = {
                        'layer': layer_name,
                        'channels': n_channels
                    }
            
            return hook

        # MaxPool hook
        def maxpool_hook(name):
            def hook(module, inputs, output):
                input_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
                
                if isinstance(input_tensor, torch.Tensor) and hasattr(input_tensor, '_source_info'):
                    n_channels = input_tensor.size(1)
                    src_layer = input_tensor._source_info['layer']
                    op_name = f"maxpool_{len(self.layer_info)}"
                    
                    # Process each channel
                    for i in range(n_channels):
                        src_key = (src_layer, 0, i)  # Source layer output
                        maxpool_key = (op_name, 0, i)  # MaxPool output
                        
                        # Connect source to maxpool function
                        self.dependencies[src_key].append(maxpool_key)
                        self.dependencies[maxpool_key].append(src_key)
                    
                    # Add source info to result
                    if isinstance(output, torch.Tensor):
                        output._source_info = {
                            'layer': op_name,
                            'channels': n_channels
                        }
                    self.layer_info[op_name] = {'type': 'maxpool', 'channels': n_channels}
                    
                return output
            return hook

        # Register hooks for all relevant layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook = module.register_forward_hook(get_layer_hook(name, module))
                self.hooks.append(hook)

    def get_dependencies(self, layer_name: str, kernel_index: int, io_type: int) -> List[Tuple[str, int, int]]:
        """Get dependencies for a specific layer and kernel index"""
        key = (layer_name, kernel_index, io_type)
        return list(set(self.dependencies[key]))  # Remove duplicates
    
    def get_layer_info(self):
        """Get information about all layers"""
        return self.layer_info
    
    def build_complete_dependencies(self):
        """Build complete dependency relationships using UnionFind"""
        # Create UnionFind instance
        uf = UnionFind()
        
        # Initialize all dimensions in the UnionFind
        all_dims = set()
        for key in self.dependencies:
            all_dims.add(key)
            uf.make_set(key)
        
        # Merge dimensions based on all existing dependencies
        for dim in all_dims:
            for dep in self.dependencies[dim]:
                uf.union(dim, dep)
        
        # Get all dimension groups
        dimension_groups = uf.get_sets()
        
        # Build complete dependency dictionary
        complete_deps = defaultdict(set)
        for group in dimension_groups:
            # For each dimension in the group
            for dim in group:
                # Add all other dimensions in the same group as dependencies
                complete_deps[dim].update(d for d in group if d != dim)
        
        return complete_deps

    def is_network_layer(self, name):
        """Check if a layer is a network layer (not an operation layer or parameter-free layer)"""
        return not (name.startswith('cat_') or 
                   name.startswith('chunk_') or 
                   name.startswith('add_') or 
                   name.startswith('relu_') or
                   name.startswith('maxpool_') or  # Add maxpool as a parameter-free layer
                   name == 'input')

    def print_dependencies(self):
        """Print structured pruning dependencies"""
        # Get complete dependency relationships
        complete_deps = self.build_complete_dependencies()
        
        # Print network layer dependencies
        print("\n=== Network Layer Dependencies ===")
        for key in sorted(complete_deps.keys()):
            if self.is_network_layer(key[0]):
                # Only output dependencies between network layers
                deps = {dep for dep in complete_deps[key] 
                       if self.is_network_layer(dep[0])}
                if deps:  # Only print if there are dependencies
                    print(f"{key}  {sorted(deps)}")
    
    def remove_hooks(self):
        """Clean up by removing hooks and restoring original operations"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Restore original operations
        torch.cat = self._original_cat
        torch.chunk = self._original_chunk
        torch.Tensor.__add__ = self._original_add
        
        # Restore activation functions
        for act_name, act_fn in self._original_activations.items():
            if act_name == 'relu':
                nn.functional.relu = act_fn
            elif act_name == 'sigmoid':
                torch.sigmoid = act_fn
            elif act_name == 'tanh':
                torch.tanh = act_fn
            elif act_name == 'leaky_relu':
                nn.functional.leaky_relu = act_fn
            elif act_name == 'elu':
                nn.functional.elu = act_fn
            elif act_name == 'selu':
                nn.functional.selu = act_fn
            elif act_name == 'gelu':
                nn.functional.gelu = act_fn
            elif act_name == 'prelu':
                nn.functional.prelu = act_fn
            elif act_name == 'softmax':
                nn.functional.softmax = act_fn
            elif act_name == 'mish':
                nn.functional.mish = act_fn
