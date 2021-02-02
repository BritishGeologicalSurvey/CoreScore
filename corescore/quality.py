"""Calculating core quality metrics from labelled fragments"""
from skimage.measure import label, regionprops
from corescore.masks import LABELS

MEASURES = ["relative_area", "fragment_perimeter",
            "average_perimeter", "total_fragments",
            "absolute_area", "perimeter_complexity"]


class QualityIndex():
    def __init__(self, mask_arr):
        self.image_mask = mask_arr
        self.labels = self.rock_labels_only(mask_arr)
        self.distributions = {}
        self.fragments = regionprops(label(self.labels))

    def rock_labels_only(self, mask):
        """
        Filter out non-rock labels from our image mask.
        'Void' is 0, everything after Rock_Fragment_2 is not rock.
        skimage.measure.label interprets 0 as background by default;
        Set everything that isn't rock to background.
        Accepts a greyscale mask with values from corescore.masks.LABELS

        """
        mask[mask > LABELS.index('Rock_Fragment_2')] = 0
        return mask

    def metric_distribution(self, attribute):
        """Returns the selected attribute for all identified fragments.
        For available attributes see skimage.measure.regionprops
        """
        return [getattr(fragment, attribute) for fragment in self.fragments]

    def absolute_area(self, min_area=None):
        """
        Return total area of all rock fragments
        """
        areas = [fragment.area for fragment in self.fragments]
        if min_area:
            list(filter(lambda x: x > min_area, areas))
        return sum(areas)

    def relative_area(self, min_area=None):
        """
        Return ratio of total fragment area to image area
        """
        areas = [fragment.area for fragment in self.fragments]
        if min_area:
            fragment_area = self.absolute_area()

        fragment_area = sum(areas)
        x, y = self.image_mask.shape
        total_area = x * y
        return fragment_area / total_area

    def fragment_perimeter(self):
        """Return total fragment perimeter"""
        return sum([fragment.perimeter for fragment in self.fragments])

    def average_perimeter(self):
        """Return average fragment perimeter"""
        return self.fragment_perimeter() / len(self.fragments)

    def total_fragments(self):
        """Return total number of fragments in image mask"""
        return len(self.fragments)

    def perimeter_complexity(self):
        """Return the ratio of total perimeter to total area"""
        return self.fragment_perimeter() / self.absolute_area()

    def parameters(self):
        """Return all the useful metrics for Core Quality Index"""
        params = {}
        for param in MEASURES:
            if(getattr(self, param)() > 1):
                params[param] = round(getattr(self, param)(), 2)
            else:
                params[param] = round(getattr(self, param)(), 5)
        return params
