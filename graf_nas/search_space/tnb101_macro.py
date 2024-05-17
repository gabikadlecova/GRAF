def tnb101_macro_encode(net):
    if isinstance(net, str):
        net = net.strip('()').split(',')

    res = []

    for idx in net:
        encoding = {}
        idx = int(idx)
        encoding['channel'] = idx % 2 == 0
        encoding['stride'] = idx > 2
        res.append(encoding)

    return res
