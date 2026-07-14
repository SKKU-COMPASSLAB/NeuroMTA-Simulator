from typing import Any, Callable
from neuromta.framework.logger import logger
from neuromta.framework.memory_handle import Pointer


__all__ = ["VariableHandle", "FIFOBufferHandle"]


class VariableHandle:
    class ActionCondition:
        EQ = "equals_to"
        GE = "greater_equal"
        LE = "less_equal"
        GT = "greater_than"
        LT = "less_than"

        def __init__(self, condition: Callable[[int], bool] | str, *args):
            if isinstance(condition, str):
                if condition == self.EQ:
                    self.condition = lambda x: x == args[0].value if isinstance(args[0], VariableHandle) else x == args[0]
                    self.action_id = f"equals_to_{args[0]}"
                elif condition == self.GE:
                    self.condition = lambda x: x >= args[0].value if isinstance(args[0], VariableHandle) else x >= args[0]
                    self.action_id = f"greater_equal_{args[0]}"
                elif condition == self.LE:
                    self.condition = lambda x: x <= args[0].value if isinstance(args[0], VariableHandle) else x <= args[0]
                    self.action_id = f"less_equal_{args[0]}"
                elif condition == self.GT:
                    self.condition = lambda x: x > args[0].value if isinstance(args[0], VariableHandle) else x > args[0]
                    self.action_id = f"greater_than_{args[0]}"
                elif condition == self.LT:
                    self.condition = lambda x: x < args[0].value if isinstance(args[0], VariableHandle) else x < args[0]
                    self.action_id = f"less_than_{args[0]}"
                else:
                    raise Exception(f"Unsupported condition string '{condition}' for ActionCondition. Supported conditions are: '{self.EQ}', '{self.GE}', '{self.LE}', '{self.GT}', and '{self.LT}'.")
            elif callable(condition):
                self.condition = condition
                self.action_id = condition.__name__ if hasattr(condition, "__name__") else id(condition)
            else:
                raise Exception(f"Condition for ActionCondition must be either a supported condition string or a callable function, but got {type(condition).__name__}.")

        def __call__(self, x: int):
            if isinstance(x, VariableHandle):
                x = x.value
            return self.condition(x)

        @property
        def signature(self) -> str:
            return self.action_id

        def __repr__(self):
            return f"ActionCondition({self.action_id})"

    def __init__(self, handle_name: str, initial_value: int=0):
        self.handle_name = handle_name

        self._value: int = initial_value
        self._value_conditional_action_methods: dict[Any, tuple[VariableHandle.ActionCondition, list[Callable]]] = {}

    def add_conditional_action(self, condition: 'VariableHandle.ActionCondition', action: Callable):
        if condition.signature not in self._value_conditional_action_methods:
            self._value_conditional_action_methods[condition.signature] = (condition, [])
        self._value_conditional_action_methods[condition.signature][1].append(action)

    def _run_actions(self):
        for signature in list(self._value_conditional_action_methods.keys()):
            condition, actions = self._value_conditional_action_methods[signature]
            if condition(self._value):
                for action in actions:
                    action()
                del self._value_conditional_action_methods[signature]

    def atomic_update(self, value: int):
        if isinstance(value, VariableHandle):
            value = value.value
        self._value = value
        self._run_actions()

    def atomic_compare_and_swap(self, cmp_value: int, new_value: int, callback: Callable = None):
        if self._value == cmp_value:
            self._value = new_value
            if callback is not None:
                callback()
            self._run_actions()
        else:
            def _action():
                self._value = new_value
                if callback is not None:
                    callback()
                self._run_actions()

            self.add_conditional_action(self.equals_to(cmp_value), _action)

    def atomic_wait(self, expected_value: int, callback: Callable):
        if self._value == expected_value:
            callback()
        else:
            self.add_conditional_action(self.equals_to(expected_value), callback)

    def atomic_wait_conditional(self, condition: 'VariableHandle.ActionCondition', callback: Callable):
        if not isinstance(condition, VariableHandle.ActionCondition):
            raise Exception(f"Condition for atomic_wait_conditional must be an instance of VariableHandle.ActionCondition, but got {type(condition).__name__}.")

        if condition(self._value):
            callback()
        else:
            self.add_conditional_action(condition, callback)

    def atomic_increase(self, increment: int, callback: Callable = None):
        self._value += increment
        if callback is not None:
            callback()
        self._run_actions()

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_value: int):
        self.atomic_update(new_value)

    def __repr__(self):
        return f"VariableHandle(name={self.handle_name}, value={self._value})"

    def __str__(self):
        return f"VariableHandle(name={self.handle_name}, value={self._value})"

    @classmethod
    def tmp(cls, initial_value: int=0) -> "VariableHandle":
        var = cls(handle_name="undefined", initial_value=initial_value)
        var.handle_name = f"var({id(var):08x})"
        return var

    def equals_to(self, value: int) -> 'VariableHandle.ActionCondition':
        # method = lambda x: x == (value.value if isinstance(value, VariableHandle) else value)
        # # method.__condition_method_name = f"equals_to_{value}"
        # setattr(method, "__condition_method_name", f"equals_to_{value}")
        # return self.ActionCondition(method)
        return self.ActionCondition(self.ActionCondition.EQ, value)

    def greater_equal(self, value: int) -> 'VariableHandle.ActionCondition':
        # method = lambda x: x >= (value.value if isinstance(value, VariableHandle) else value)
        # # method.__condition_method_name = f"greater_equal_{value}"
        # setattr(method, "__condition_method_name", f"greater_equal_{value}")
        # return self.ActionCondition(method)
        return self.ActionCondition(self.ActionCondition.GE, value)

    def less_equal(self, value: int) -> 'VariableHandle.ActionCondition':
        # method = lambda x: x <= (value.value if isinstance(value, VariableHandle) else value)
        # # method.__condition_method_name = f"less_equal_{value}"
        # setattr(method, "__condition_method_name", f"less_equal_{value}")
        # return self.ActionCondition(method)
        return self.ActionCondition(self.ActionCondition.LE, value)

    def greater_than(self, value: int) -> 'VariableHandle.ActionCondition':
        # method = lambda x: x > (value.value if isinstance(value, VariableHandle) else value)
        # # method.__condition_method_name = f"greater_than_{value}"
        # setattr(method, "__condition_method_name", f"greater_than_{value}")
        # return self.ActionCondition(method)
        return self.ActionCondition(self.ActionCondition.GT, value)

    def less_than(self, value: int) -> 'VariableHandle.ActionCondition':
        # method = lambda x: x < (value.value if isinstance(value, VariableHandle) else value)
        # # method.__condition_method_name = f"less_than_{value}"
        # setattr(method, "__condition_method_name", f"less_than_{value}")
        # return self.ActionCondition(method)
        return self.ActionCondition(self.ActionCondition.LT, value)


class FIFOBufferHandle:
    def __init__(self, handle_name: str, depth: int, entry_size: int):
        self.handle_name = handle_name
        self.depth = depth
        self.entry_size = entry_size
        self._mem_ptr: Pointer = Pointer()  # empty pointer, to be allocated by the core

        self._ref_counts: list[VariableHandle] = [VariableHandle(f"{handle_name}_ref_counter_{i}", initial_value=0) for i in range(depth)]
        self._global_counter: VariableHandle = VariableHandle(f"{handle_name}_global_counter", initial_value=0)
        self._slot_entry_tags: list[int] = [-1 for _ in range(depth)]

        self._entry_vacant_action_methods: dict[tuple[int, ...], list[Callable]] = {}
        self._entry_valid_action_methods: dict[tuple[int, ...], list[Callable]] = {}

    @staticmethod
    def _normalize_entry_ids(entry_ids: int | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(entry_ids, int):
            entry_ids = (entry_ids,)
        elif isinstance(entry_ids, (list, tuple)):
            entry_ids = tuple(entry_ids)
        else:
            raise Exception(f"FIFO entry id must be an int or a sequence of ints, but got {type(entry_ids).__name__}.")

        for entry_id in entry_ids:
            if not isinstance(entry_id, int):
                raise Exception(f"FIFO entry id must be an int, but got {type(entry_id).__name__}.")
            if entry_id < 0:
                raise Exception(f"FIFO entry id must be non-negative, but got {entry_id}.")
        return entry_ids

    def _check_batch_slot_conflict(self, entry_ids: tuple[int, ...]):
        slot_entries: dict[int, int] = {}
        for entry_id in entry_ids:
            entry_idx = entry_id % self.depth
            if entry_idx in slot_entries and slot_entries[entry_idx] != entry_id:
                raise Exception(
                    f"FIFO buffer '{self.handle_name}' cannot access entries {slot_entries[entry_idx]} and {entry_id} "
                    f"in the same burst because they map to the same physical slot {entry_idx}."
                )
            slot_entries[entry_idx] = entry_id

    def _run_actions(self):
        for entry_ids in list(self._entry_vacant_action_methods.keys()):
            if self._entries_vacant_condition(self, entry_ids):
                for action in self._entry_vacant_action_methods[entry_ids]:
                    action()
                del self._entry_vacant_action_methods[entry_ids]

        for entry_ids in list(self._entry_valid_action_methods.keys()):
            if self._entries_valid_condition(self, entry_ids):
                for action in self._entry_valid_action_methods[entry_ids]:
                    action()
                del self._entry_valid_action_methods[entry_ids]

    @staticmethod
    def _entry_vacant_condition(buf: 'FIFOBufferHandle', entry_id: int) -> bool:
        entry_idx = entry_id % buf.depth
        return buf._ref_counts[entry_idx].value == 0

    @staticmethod
    def _entry_valid_condition(buf: 'FIFOBufferHandle', entry_id: int) -> bool:
        entry_idx = entry_id % buf.depth
        if buf._slot_entry_tags[entry_idx] != entry_id:
            return False
        return buf._ref_counts[entry_idx].value > 0

    @staticmethod
    def _entries_vacant_condition(buf: 'FIFOBufferHandle', entry_ids: tuple[int, ...]) -> bool:
        buf._check_batch_slot_conflict(entry_ids)
        return all(buf._entry_vacant_condition(buf, entry_id) for entry_id in set(entry_ids))

    @staticmethod
    def _entries_valid_condition(buf: 'FIFOBufferHandle', entry_ids: tuple[int, ...]) -> bool:
        buf._check_batch_slot_conflict(entry_ids)
        return all(buf._entry_valid_condition(buf, entry_id) for entry_id in set(entry_ids))

    def wait_until_vacant(self, entry_ids: int | list[int] | tuple[int, ...], callback: Callable):
        entry_ids = self._normalize_entry_ids(entry_ids)
        self._run_actions()

        if self._entries_vacant_condition(self, entry_ids):
            callback()
        else:
            if entry_ids not in self._entry_vacant_action_methods:
                self._entry_vacant_action_methods[entry_ids] = []
            self._entry_vacant_action_methods[entry_ids].append(callback)

    def wait_until_valid(self, entry_ids: int | list[int] | tuple[int, ...], callback: Callable):
        entry_ids = self._normalize_entry_ids(entry_ids)
        self._run_actions()

        if self._entries_valid_condition(self, entry_ids):
            callback()
        else:
            if entry_ids not in self._entry_valid_action_methods:
                self._entry_valid_action_methods[entry_ids] = []
            self._entry_valid_action_methods[entry_ids].append(callback)

    def write_entry(self, entry_id: int, ref_count: int):
        if not self._entry_vacant_condition(self, entry_id):
            raise Exception(f"Attempting to write to a non-vacant entry (entry_id={entry_id}) in FIFO buffer '{self.handle_name}'.")
        if ref_count <= 0:
            raise Exception(f"FIFO entry ref_count must be positive, but got {ref_count}.")

        entry_idx = entry_id % self.depth
        self._slot_entry_tags[entry_idx] = entry_id
        self._ref_counts[entry_idx].value = ref_count
        self._global_counter.atomic_update(max(entry_id + 1, self._global_counter.value))

        self._run_actions()

    def write_entries(self, entry_refs: list[tuple[int, int]] | tuple[tuple[int, int], ...]):
        entry_ids = tuple(entry_id for entry_id, _ in entry_refs)
        self._check_batch_slot_conflict(entry_ids)
        if not self._entries_vacant_condition(self, entry_ids):
            raise Exception(f"Attempting to write to non-vacant entries {entry_ids} in FIFO buffer '{self.handle_name}'.")

        for entry_id, ref_count in entry_refs:
            if ref_count <= 0:
                raise Exception(f"FIFO entry ref_count must be positive, but got {ref_count}.")
            entry_idx = entry_id % self.depth
            self._slot_entry_tags[entry_idx] = entry_id
            self._ref_counts[entry_idx].value = ref_count
            self._global_counter.atomic_update(max(entry_id + 1, self._global_counter.value))

        self._run_actions()

    def read_entry(self, entry_id: int, count: int=1):
        if not self._entry_valid_condition(self, entry_id):
            raise Exception(f"Attempting to read from a non-valid entry (entry_id={entry_id}) in FIFO buffer '{self.handle_name}'.")
        if count <= 0:
            raise Exception(f"FIFO entry read count must be positive, but got {count}.")

        entry_idx = entry_id % self.depth
        if self._ref_counts[entry_idx].value < count:
            raise Exception(
                f"Attempting to read FIFO entry {entry_id} in buffer '{self.handle_name}' {count} times, "
                f"but only {self._ref_counts[entry_idx].value} references remain."
            )
        self._ref_counts[entry_idx].atomic_increase(-count)
        if self._ref_counts[entry_idx].value == 0:
            self._slot_entry_tags[entry_idx] = -1

        self._run_actions()

    def read_entries(self, entry_ids: list[int] | tuple[int, ...]):
        entry_ids = self._normalize_entry_ids(entry_ids)
        if not self._entries_valid_condition(self, entry_ids):
            raise Exception(f"Attempting to read from non-valid entries {entry_ids} in FIFO buffer '{self.handle_name}'.")

        for entry_id in entry_ids:
            self.read_entry(entry_id)

    @property
    def mem_ptr(self) -> Pointer:
        return self._mem_ptr

    def get_ptr(self, entry_id: int) -> Pointer:
        entry_id = entry_id % self.depth
        return Pointer(self.mem_ptr.addr + entry_id * self.entry_size)

    def get_entry_idx(self, ptr: Pointer) -> int:
        entry_idx = (ptr.addr - self.mem_ptr.addr) // self.entry_size
        if entry_idx < 0 or entry_idx >= self.depth:
            raise Exception(f"Pointer {ptr} is out of range for the FIFO buffer '{self.handle_name}'.")
        return entry_idx

    def get_ref_count(self, entry_id: int) -> VariableHandle:
        entry_id = entry_id % self.depth
        return self._ref_counts[entry_id]

    def __repr__(self):
        # valid_slots = [i for i in range(self.depth) if self._ref_counts[i].value > 0]
        slot_info = [f"slot[{i}]: (ref_count={self._ref_counts[i].value}, tag={self._slot_entry_tags[i]})" for i in range(self.depth)]
        return f"FIFOBufferHandle(name={self.handle_name}, depth={self.depth}, entry_size={self.entry_size}, slot_info=[{', '.join(slot_info)}])"
