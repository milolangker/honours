from __future__ import annotations
import jax.numpy as np
from jax import Array, vmap
import dLux
import os #ADDED BY ME (MILO)
import dLux.utils as dlu
from jax.scipy.ndimage import map_coordinates

__all__ = ["ApplyBasisCLIMB", "TolimanPupilLayer", "PhaseGratingLayer", "TolimanApertureLayer"]

OpticalLayer = lambda: dLux.optical_layers.OpticalLayer
BasisLayer = lambda: dLux.optical_layers.BasisLayer

# ADDED BY ME (MILO)
AberratedLayer = lambda: dLux.optical_layers.AberratedLayer
TransmissiveLayer = lambda: dLux.optical_layers.TransmissiveLayer

# TODO write this class: make it take arbitrary oversamples (its hardcoded to 3x),
#  and take arbitrary output size (its hardcoded to 256)
class ApplyBasisCLIMB(BasisLayer()):
    """
    Adds an array of binary phase values to the input wavefront from a set of
    continuous basis vectors. This uses the CLIMB algorithm in order to
    generate the binary values in a continuous manner as described in the
    paper Wong et al. 2021. The basis vectors are taken as an Optical Path
    Difference (OPD), and applied to the phase of the wavefront. The ideal
    wavelength parameter described the wavelength that will have a perfect
    anti-phase relationship given by the Optical Path Difference.

    TODO: Many of the methods in the class still need doccumentation.
    TODO: This currently only outputs 256 pixel arrays and uses a 3x oversample,
    therefore requiring a 768 pixel basis array.

    "This class is the crazy lady we keep in the attic, but turns out her name is actually Ben." - L. Desdoigts, 2023.

    Attributes
    ----------
    basis: Array
        Arrays holding the continous pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    ideal_wavelength : Array
        The target wavelength at which a perfect anti-phase relationship is
        applied via the OPD.
    """

    # basis            : Array
    # coefficients     : Array
    ideal_wavelength: Array

    def __init__(
        self: OpticalLayer(),
        basis: Array,
        ideal_wavelength: Array,
        coefficients: Array = None,
    ) -> OpticalLayer():
        """
        Constructor for the ApplyBasisCLIMB class.

        Parameters
        ----------
        basis : Array
            Arrays holding the continous pre-calculated basis vectors. This must
            be a 3d array of shape (nterms, npixels, npixels), with the final
            two dimensions matching that of the wavefront at time of
            application. This is currently required to be a nx768x768 shaped
            array.
        ideal_wavelength : Array
            The target wavelength at which a perfect anti-phase relationship is
            applied via the OPD.
        coefficients : Array = None
            The Array of coefficients to be applied to each basis vector. This
            must be a one dimensional array with leading dimension equal to the
            leading dimension of the basis vectors. Default is None which
            initialises an array of zeros.
        """
        super().__init__(basis=basis, coefficients=coefficients)
        # self.basis            = np.asarray(basis, dtype=float)
        self.ideal_wavelength = np.asarray(ideal_wavelength, dtype=float)
        # self.coefficients     = np.array(coefficients).astype(float) \
        #             if coefficients is not None else np.zeros(len(self.basis))

        # # Inputs checks
        # assert self.basis.ndim == 3, \
        # ("basis must be a 3 dimensional array, ie (nterms, npixels, npixels).")
        # assert self.basis.shape[-1] == 768, \
        # ("Basis must have shape (n, 768, 768).")
        # assert self.coefficients.ndim == 1 and \
        # self.coefficients.shape[0] == self.basis.shape[0], \
        # ("coefficients must be a 1 dimensional array with length equal to the "
        # "First dimension of the basis array.")
        # assert self.ideal_wavelength.ndim == 0, ("ideal_wavelength must be a "
        #                                          "scalar array.")

    # def __call__(self: OpticalLayer(), wavefront: Wavefront) -> Wavefront:
    def apply(self: OpticalLayer(), wavefront: Wavefront) -> Wavefront:
        """
        Generates and applies the binary OPD array to the wavefront in a
        differentiable manner.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the binary OPD applied.
        """
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi * self.CLIMB(latent, ppsz=wavefront.npixels)
        opd = self.phase_to_opd(binary_phase, self.ideal_wavelength)
        return wavefront.add_opd(opd)

    @property
    def applied_shape(self):
        return tuple(np.array(self.basis.shape[-2:]) // 3)

    def opd_to_phase(self, opd, wavel):
        return 2 * np.pi * opd / wavel

    def phase_to_opd(self, phase, wavel):
        return phase * wavel / (2 * np.pi)

    def get_opd(self, basis, coefficients):
        return np.dot(basis.T, coefficients)

    def get_total_opd(self):
        return self.get_opd(self.basis, self.coefficients)

    def get_binary_phase(self):
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi * self.CLIMB(latent)
        return binary_phase

    def lsq_params(self, img):
        xx, yy = np.meshgrid(
            np.linspace(0, 1, img.shape[0]), np.linspace(0, 1, img.shape[1])
        )
        A = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]).T
        matrix = np.linalg.inv(np.dot(A.T, A)).dot(A.T)
        return matrix, xx, yy, A

    def lsq(self, img):
        matrix, _, _, _ = self.lsq_params(img)
        return np.dot(matrix, img.ravel())

    def area(self, img, epsilon=1e-15):
        a, b, c = self.lsq(img)
        a = np.where(a == 0, epsilon, a)
        b = np.where(b == 0, epsilon, b)
        c = np.where(c == 0, epsilon, c)
        x1 = (-b - c) / (a)  # don't divide by zero
        x2 = -c / (a)  # don't divide by zero
        x1, x2 = np.min(np.array([x1, x2])), np.max(np.array([x1, x2]))
        x1, x2 = np.max(np.array([x1, 0])), np.min(np.array([x2, 1]))

        dummy = (
            x1
            + (-c / b) * x2
            - (0.5 * a / b) * x2**2
            - (-c / b) * x1
            + (0.5 * a / b) * x1**2
        )

        # Set the regions where there is a defined gradient
        dummy = np.where(dummy >= 0.5, dummy, 1 - dummy)

        # Colour in regions
        dummy = np.where(np.mean(img) >= 0, dummy, 1 - dummy)

        # rescale between 0 and 1?
        dummy = np.where(np.all(img > 0), 1, dummy)
        dummy = np.where(np.all(img <= 0), 0, dummy)

        # undecided region
        dummy = np.where(np.any(img == 0), np.mean(dummy > 0), dummy)

        # rescale between 0 and 1
        dummy = np.clip(dummy, 0, 1)

        return dummy

    def CLIMB(self, wf, ppsz=256):
        psz = ppsz * 3
        dummy = np.array(np.split(wf, ppsz))
        dummy = np.array(np.split(np.array(dummy), ppsz, axis=2))
        subarray = dummy[:, :, 0, 0]

        flat = dummy.reshape(-1, 3, 3)
        vmap_mask = vmap(self.area, in_axes=(0))

        soft_bin = vmap_mask(flat).reshape(ppsz, ppsz)

        return soft_bin

# NEW STUFF ADDED BY ME (Milo)
class TolimanPupilLayer(AberratedLayer()):
    def __init__(
        self, 
        wf_npixels: int
    ):
        """
        Creates the toliman pupil layer.

        Parameters
        ----------
        wf_npixels : int
            The resolution of the pupil/aperture. Should be a power of 2.

        Returns
        -------
        The correct Toliman Pupil layer.
        """

        # Put this here for now, will be in dLux eventually
        def scale_array(array: Array, size_out: int, order: int) -> Array:
            xs = np.linspace(0, array.shape[0], size_out)
            xs, ys = np.meshgrid(xs, xs)
            return map_coordinates(array, np.array([ys, xs]), order=order)
        
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diffractive_pupil.npy")

        mask = scale_array(np.load(path), wf_npixels, order=1)

        # Enforce full binary
        mask = mask.at[np.where(mask <= 0.5)].set(0.0)
        mask = mask.at[np.where(mask > 0.5)].set(1.0)

        # Multiply by pi
        pupil = mask * np.pi

        # Initialize the AberratedLayer with the computed phase
        super().__init__(phase=pupil)

class PhaseGratingLayer(AberratedLayer()):
    def __init__(
        self, 
        wf_npixels: int, 
        diameter: float, 
        period: float, 
        phase_difference: float, 
        toliman: bool = False
    ):
        """
        Creates the phase grating layer.
        
        Parameters
        ----------
        wf_npixels : int
            Resolution of the pupil/aperture. Should be a power of 2.

        diameter : float
            The diameter of the pupil/aperture in meters.

        period: float
            The period of the phase grating in meters.

        phase_difference: float
            The maximum phase difference in radians.

        toliman: bool = False
            Whether to apply the inversion of the phase grating necessary for the toliman pupil.
            Set to true when constructing a toliman pupil, false for regular aperture.

        Returns
        -------
        The correct phase grating layer.
        """
        # NEED COORDS FOR GRATING
        x = np.arange(wf_npixels)
        y = np.arange(wf_npixels)
        X, Y = np.meshgrid(x,y, indexing = 'ij') # I believe the indexing part is neccessary? Anyway doesn't break anything

        # converted to pixels
        gratingPeriodPixels = period * (wf_npixels/diameter)

        # note that the magic of the pupil allows it to apply the same phase across all wavelengths!
        # amplitude = phase_difference / 2
        grating = phase_difference/2 * np.sin((X + Y) * 2 * np.pi / gratingPeriodPixels) #first part
        grating += phase_difference/2 * np.sin((X - Y) * 2 * np.pi / gratingPeriodPixels) #second part

        # because we add two, one for each direction
        grating /= 2
        # TODO: add initial grating offsets + angle?

        # For toliman pupil
        if toliman == True:
            # Construct the file path (relative to the script's location)
            # Put this here for now, will be in dLux eventually
            def scale_array(array: Array, size_out: int, order: int) -> Array:
                xs = np.linspace(0, array.shape[0], size_out)
                xs, ys = np.meshgrid(xs, xs)
                return map_coordinates(array, np.array([ys, xs]), order=order)
            
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diffractive_pupil.npy")

            mask = scale_array(np.load(path), wf_npixels, order=1)

            # Enforce full binary
            mask = mask.at[np.where(mask <= 0.5)].set(0.0)
            mask = mask.at[np.where(mask > 0.5)].set(1.0)

            # this shit is whack. Just multiplies by -1 when necessary.
            grating = grating.at[mask > np.max(mask)/2].set(-1*grating[mask > np.max(mask)/2])

        super().__init__(phase=grating)

class TolimanApertureLayer(TransmissiveLayer()):
    def __init__(
        self,
        wf_npixels: int,
        m1_diameter: float = 0.125,
        m2_diameter: float = 0.032,
        strut_width: float = 0.002
    ):
        """
        Creates the toliman aperture layer.
        
        Parameters
        ----------
        wf_npixels : int
            Resolution of the pupil/aperture. Should be a power of 2.

        m1_diameter : float
            The diameter of the primary mirror in meters.

        m2_diamter: float
            The diameter of the secondary mirror in meters.

        strut_width: float
            The width of the supporting struts in meters.

        Returns
        -------
        The correct toliman aperture layer.
        """
        diameter = m1_diameter

        # Generate Aperture
        coords = dlu.pixel_coords(5 * wf_npixels, diameter)
        outer = dlu.circle(coords, m1_diameter / 2)
        inner = dlu.circle(coords, m2_diameter / 2, invert=True)
        spiders = dlu.spider(coords, strut_width, [0, 120, 240])
        transmission = dlu.combine([outer, inner, spiders], 5)

        super().__init__(transmission=transmission)

# Gonna make the 'sidelobe shift layer'
class SidelobeShiftLayer(AberratedLayer()):
    def __init__(
        self,
        wf_npixels: int,
        diameter: float,
        period: float,
        central_wavelength: float,
        psf_pixel_scale: float,
        oversample: int,
        corner: Array = np.array([-1,-1]), # bottom left corner
    ):
        """
        Creates a layer to shift the psf such that
        the appropriate sidelobe is in the centre.
        
        Parameters
        ----------
        wf_npixels : int
            Resolution of the pupil/aperture. Should be a power of 2.

        diameter : float
            The diameter of aperture in meters.

        period: float
            The period of the phase grating in meters.

        central_wavelength: float
            The wavelength of the sidelobe to centre on, in meters.

        corner: Array
            The corner to simulate. 
            [-1, -1] = bottom left, [1, -1] = bottom right, 
            [-1,  1] = top left,    [1,  1] = top right.

        psf_pixel_scale: float
            The pixel scale of the psf simulated in arcseconds per pixel. 
            Necessary to ensure the correct offset.

        oversample: int
            The oversampling factor of the PSF.
            Necessary to ensure the correct offset.
        
        Returns
        -------
        The correct phase ramp to simuate a sidelobe.
        """  

        # angle that central wavelength makes with phase grating
        angle = np.arcsin(central_wavelength/period)

        # true pixel scale (radians per pixel)
        true_pixel_scale = dlu.arcsec2rad(psf_pixel_scale/oversample)

        # central offset (pixels, integer) --> radians
        centre_offset = (angle // true_pixel_scale) * true_pixel_scale

        coords = dlu.pixel_coords(wf_npixels, diameter)

        # phase shift
        sidelobe_shift = (2 * np.pi / central_wavelength) * (coords[0] * corner[0] * np.sin(centre_offset) + coords[1] * corner[1] * np.sin(centre_offset)) 
        # works for corners specified.
        # could do % 2 * np.pi, but found that for whatever reason made it (slightly) worse
        # also... errors found to be slightly better if you just don't even have np.sin (i.e. x = sinx)
        # SUS!!

        super().__init__(phase=sidelobe_shift)