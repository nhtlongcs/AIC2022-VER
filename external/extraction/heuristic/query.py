from .caption import Caption
from ..utils.common import refine_list_colors, refine_list_subjects
from typing import Optional


class Query(object):
    def __init__(
        self, query_content: dict, query_id: str, query_order: Optional[str] = None
    ):
        self.query_id = query_id
        self.query_order = query_order
        self._setup(query_content)

    def _setup(self, query_content):
        # Init captions
        self.list_caps = []
        for cap_id in query_content.keys():
            self.list_caps.append(Caption(query_content[cap_id], cap_id))

        # Find subject
        self.subjects = [cap.main_subject for cap in self.list_caps]
        self.objects = [svo["O"] for svo in self.get_all_SVO_info()]

        self._get_list_colors()
        self._refine_subjects()
        # self._refine_colors()

        self._get_list_action()
        self._refine_list_action()

    def _get_list_action(self):
        self.actions = []
        for cap in self.list_caps:
            if len(cap.sv_format) == 0:
                continue
            for sv in cap.sv_format:
                self.actions.append(sv["V"])

        pass

    def _refine_list_action(self, unique=True):
        if unique:
            self.actions = list(set(self.actions))

        # self.actions = remove_redundant_actions(self.actions)
        pass

    def _refine_colors(self, unique=True):
        self.colors = refine_list_colors(self.colors, unique)
        pass

    def _refine_subjects(self, unique=True):
        """Add rules to refine vehicle list for the given query"""
        self.subjects = refine_list_subjects(self.subjects, unique)

    def get_list_captions_str(self):
        list_cap_str = [c.caption for c in self.list_caps]
        return "\n".join(list_cap_str)

    def _get_list_colors(self):
        self.colors = []
        for cap in self.list_caps:
            self.colors.extend(list(set(cap.subject.combines)))

    def get_all_SV_info(self):
        sv_samples = []
        for cap in self.list_caps:
            for sv in cap.sv_format:
                sv_samples.append(sv)

        return sv_samples

    def get_all_SVO_info(self):
        svo_samples = []
        for cap in self.list_caps:
            for svo in cap.svo_format:
                svo_samples.append(svo)

        return svo_samples
