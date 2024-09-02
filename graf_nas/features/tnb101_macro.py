from graf_nas.search_space.tnb101_macro import TNB101MacroGraph
from typing import Dict


def count_switches(net: TNB101MacroGraph) -> Dict[str, int]:
    op_counts = {}

    def get_key(channel, stride):
        return f"ch{channel}s{stride}"

    for channel in [True, False]:
        for stride in [True, False]:
            op_counts[get_key(channel, stride)] = 0

    for op in net:
        op_counts[get_key(op['channel'], op['stride'])] += 1

    return op_counts


def get_state_at_i(net: TNB101MacroGraph) -> Dict[str, int]:
    channel = 0
    stride = 0

    res = {}
    for i in range(6):
        if i >= len(net.ops):
            channel = 0
            stride = 0
        else:
            channel += int(net.ops[i]['channel'])
            stride += int(net.ops[i]['stride'])

        res[f"ch_pos_{i}"] = channel
        res[f"s_pos_{i}"] = stride

    return res


feature_func_dict = {
    'count_ops': count_switches,
    'pos_state': get_state_at_i
}
