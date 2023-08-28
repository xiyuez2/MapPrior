import os, hashlib
import requests
from tqdm import tqdm
import numpy as np

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


def vis_layer_6(pred):
#     map_classes:
#   - drivable_area
#   - ped_crossing
#   - walkway
#   - stop_line
#   - carpark_area
#   - divider
    # pred: shape 6,w,h
    # out: img to be visilized by plt.imshow of shape w,h,3
    try:
        pred = pred.cpu().numpy()
    except:
        pass

    # print("debug:",np.shape(pred))
    w,h = np.shape(pred)[-2],np.shape(pred)[-1]
    color = np.array([(166, 206, 227),(251, 154, 153),(227, 26, 28),(253, 191, 111),(255, 127, 0),(106, 61, 154)])
    array_to_print = np.zeros((np.shape(pred)[1],np.shape(pred)[2],3)) + 255
    thrs = [0.5]*len(pred) 
    # name = ['pedestrians','cars','road_segment', 'lane','drivable_area','ped_crossing','walkway','stop_line','carpark_area']
    order =[0,1,2,3,4,5]
    for layer_idx in order:
        ## init varibles
        cur_pred = pred[layer_idx]
        cur_color = color[layer_idx]
        cur_thr = thrs[layer_idx]
        bin_pred = cur_pred > cur_thr
        
        #clear previous layers if necessary
        clear_mask = (bin_pred == False).astype(int)
        
        cleared_array = array_to_print * np.array([clear_mask,clear_mask,clear_mask]).transpose(1,2,0)
        # add in newthings
        bin_pred = np.array([bin_pred,bin_pred,bin_pred]).transpose(1,2,0)
        array_to_print = cleared_array + bin_pred * cur_color

    # ego car
    # array_to_print[w//2-10:w//2+10,h//2-5:h//2+5,:] = np.array([50,50,100])
    return array_to_print.astype(int)

if __name__ == "__main__":
    config = {"keya": "a",
              "keyb": "b",
              "keyc":
                  {"cc1": 1,
                   "cc2": 2,
                   }
              }
    from omegaconf import OmegaConf
    config = OmegaConf.create(config)
    print(config)
    retrieve(config, "keya")

