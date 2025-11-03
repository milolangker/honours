from __future__ import annotations
import jax.numpy as np
from jax import Array, vmap
from jax.scipy.ndimage import map_coordinates
import dLux.utils as dlu
import dLux.layers as dll
import dLux
import dLuxToliman
import os

MixedAlphaCen = lambda: dLuxToliman.sources.MixedAlphaCen

__all__ = ["TolimanOpticalSystem", "SideLobeSystem", "SideLobeTelescope", "SideLobeCLIMB"]

OpticalLayer = lambda: dLux.optical_layers.OpticalLayer
AngularOpticalSystem = lambda: dLux.optical_systems.AngularOpticalSystem
Telescope = lambda: dLux.instruments.Telescope

# no need for lambda
PointSource = dLux.sources.PointSource
PointSources = dLux.sources.PointSources
AngularOpticalSystem_object = dLux.optical_systems.AngularOpticalSystem
Scene = dLux.sources.Scene
from dLuxToliman.sources import AlphaCen
from dLuxToliman.sources import LobePointSource

class TolimanOpticalSystem(AngularOpticalSystem()):
    def __init__(
        self,
        wf_npixels: int = 256,
        psf_npixels: int = 128,
        oversample: int = 4,
        psf_pixel_scale: float = 0.375,  # arcsec
        mask: Array = None,
        radial_orders: Array = None,
        noll_indices: Array = None,
        coefficients: Array = None,
        m1_diameter: float = 0.125,
        m2_diameter: float = 0.032,
        # n_struts: int = 3,
        strut_width: float = 0.002,
        # strut_rotation: float = -np.pi / 2,
    ):
        """
        A pre-built dLux optics layer of the Toliman optical system. Note TolimanOptics uses units of arcseconds.

        Parameters
        ----------
        wf_npixels : int
            The pixel width the wavefront layer.
        psf_npixels : int
            The pixel width of the PSF.
        oversample : int
            The Nyquist oversampling factor of the PSF.
        psf_pixel_scale : float
            The pixel scale of the PSF in arcseconds per pixel.
        mask : Array
            The diffractive mask array to apply to the wavefront layer.
        radial_orders : Array = None
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indices argument should be used instead.
        noll_indices : Array
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus.
        coefficients : Array
            The coefficients of the Zernike polynomials.
        m1_diameter : float
            The outer diameter of the primary mirror in metres.
        m2_diameter : float
            The diameter of the secondary mirror in metres.
        n_struts : int
            The number of uniformly spaced struts holding the secondary mirror.
        strut_width : float
            The width of the struts in metres.
        strut_rotation : float
            The angular rotation of the struts in radians.
        """

        # Diameter
        diameter = m1_diameter

        # Generate Aperture
        coords = dlu.pixel_coords(5 * wf_npixels, diameter)
        outer = dlu.circle(coords, m1_diameter / 2)
        inner = dlu.circle(coords, m2_diameter / 2, invert=True)
        spiders = dlu.spider(coords, strut_width, [0, 120, 240])
        transmission = dlu.combine([outer, inner, spiders], 5)

        # Hack this in for now, will be in dLux eventually
        if radial_orders is not None:
            radial_orders = np.array(radial_orders)

            if (radial_orders < 0).any():
                raise ValueError("Radial orders must be >= 0")

            noll_indices = []
            for order in radial_orders:
                start = dlu.triangular_number(order)
                stop = dlu.triangular_number(order + 1)
                noll_indices.append(np.arange(start, stop) + 1)
            noll_indices = np.concatenate(noll_indices).astype(int)

        if noll_indices is None:
            aperture = dll.TransmissiveLayer(transmission, normalise=True)
        else:
            # Generate Basis
            coords = dlu.pixel_coords(wf_npixels, diameter)
            basis = np.array(
                [dlu.zernike(i, coords, m1_diameter) for i in noll_indices]
            )

            if coefficients is None:
                coefficients = np.zeros(len(noll_indices))

            # Combine into BasisOptic class
            aperture = dll.BasisOptic(basis, transmission, coefficients, normalise=True)

        # # Generate Aperture
        # aperture = dLux.apertures.ApertureFactory(
        #     npixels=wf_npixels,
        #     radial_orders=radial_orders,
        #     noll_indices=noll_indices,
        #     coefficients=coefficients,
        #     secondary_ratio=m2_diameter / m1_diameter,
        #     nstruts=n_struts,
        #     strut_ratio=strut_width / m1_diameter,
        #     strut_rotation=strut_rotation,
        # )

        # Put this here for now, will be in dLux eventually
        def scale_array(array: Array, size_out: int, order: int) -> Array:
            xs = np.linspace(0, array.shape[0], size_out)
            xs, ys = np.meshgrid(xs, xs)
            return map_coordinates(array, np.array([ys, xs]), order=order)

        # Generate Mask
        if mask is None:
            path = os.path.join(os.path.dirname(__file__), "diffractive_pupil.npy")
            # arr_in = np.load(path)
            # ratio = wf_npixels / arr_in.shape[-1]
            mask = scale_array(np.load(path), wf_npixels, order=1)

            # Enforce full binary
            mask = mask.at[np.where(mask <= 0.5)].set(0.0)
            mask = mask.at[np.where(mask > 0.5)].set(1.0)

            # Enforce full binary
            mask = dlu.phase2opd(mask * np.pi, 585e-9)

            # Turn into optic
            mask = dll.AberratedLayer(mask)

        layers = [("aperture", aperture), ("pupil", mask)]

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            # aperture=aperture,
            # mask=mask,
            psf_npixels=int(psf_npixels),
            oversample=int(oversample),
            # psf_pixel_scale = float(psf_pixel_scale)
            psf_pixel_scale=psf_pixel_scale,
        )

    def _apply_aperture(self, wavelength, offset):
        """
        Overwrite so mask can be stored as array
        """
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.aperture
        wf = wf.normalise()
        wf += self.mask
        return wf


class TolimanSpikes(TolimanOpticalSystem):
    """
    A pre-built dLux optics layer of the Toliman optical system with diffraction spikes.

    Attributes
    ----------
    grating_depth : float
        The depth of the grating in nanometres.
    grating_period : float
        The period of the grating in microns.
    spike_npixels : int
        The pixel width of the diffraction spikes.
    """

    grating_depth: float
    grating_period: float
    spike_npixels: int

    def __init__(
        self,
        wf_npixels=256,
        psf_npixels=256,
        oversample=2,
        psf_pixel_scale=0.375,  # arcsec
        spike_npixels=512,
        mask=None,
        radial_orders: Array = None,
        noll_indices: Array = None,
        coefficients: Array = None,
        m1_diameter: float = 0.125,
        m2_diameter: float = 0.032,
        #n_struts=3, #MODIFIED BY ME
        strut_width=0.002,
        #strut_rotation=-np.pi / 2, #MODIFIED BY ME
        grating_depth=100.0,  # nm
        grating_period=300,  # um
    ):
        """
        A pre-built dLux optics layer of the Toliman optical system with diffraction spikes.

        Parameters
        ----------
        wf_npixels : int
            The pixel width the wavefront layer.
        psf_npixels : int
            The pixel width of the PSF.
        oversample : int
            The Nyquist oversampling factor of the PSF.
        psf_pixel_scale : float
            The pixel scale of the PSF in arcseconds per pixel.
        mask : Array
            The diffractive mask array to apply to the wavefront layer.
        radial_orders : Array = None
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indices argument should be used instead.
        noll_indices : Array
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus.
        coefficients : Array
            The coefficients of the Zernike polynomials.
        m1_diameter : float
            The outer diameter of the primary mirror in metres.
        m2_diameter : float
            The diameter of the secondary mirror in metres.
        n_struts : int
            The number of uniformly spaced struts holding the secondary mirror.
        strut_width : float
            The width of the struts in metres.
        strut_rotation : float
            The angular rotation of the struts in radians.
        """

        # Diameter
        self.grating_depth = grating_depth
        self.grating_period = grating_period
        self.spike_npixels = spike_npixels

        super().__init__(
            wf_npixels=wf_npixels,
            psf_npixels=psf_npixels,
            oversample=oversample,
            psf_pixel_scale=psf_pixel_scale,
            mask=mask,
            radial_orders=radial_orders,
            noll_indices=noll_indices,
            coefficients=coefficients,
            m1_diameter=m1_diameter,
            m2_diameter=m2_diameter,
            #n_struts=n_struts, # MODIFIED BY ME
            strut_width=strut_width,
            #strut_rotation=strut_rotation, # MODIFIED BY ME
        )

    def model_spike(self, wavelengths, offset, weights, angles, sign, centre):
        """
        Model a Toliman diffraction spike.
        """
        propagator = vmap(self.model_spike_mono, (0, None, 0, None, None))
        psfs = propagator(wavelengths, offset, angles, sign, centre)
        psfs *= weights[..., None, None]
        return psfs.sum(0)

    def model_spike_mono(self, wavelength, offset, angle, sign, centre):
        """
        Model a monochromatic Toliman diffraction spike.
        "Yeah most of that code was hacked together trying to not burn my legs running it." - L. Desdoigts, 2023.

        Parameters
        ----------
        wavelength : float
            The wavelength of the monochromatic PSF.
        offset : float
            Stellar positional offset from the optical axis in arcseconds. TODO CHECK
        angle : float
            The diffraction angle in radians between the spike and the star, determined by the grating period.
        sign : tuple
            Determine which corner of the detector the spike is in. E.g. [-1, 1].
        centre : tuple
            The central location of the diffraction spike in pixels.
        """

        # Construct and tilt
        wf = dLux.wavefronts.Wavefront(
            self.aperture.transmission.shape[-1], self.diameter, wavelength #MODIFIED BY ME
        )

        # Addd offset and tilt
        wf = wf.tilt(offset - sign * angle)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply aberrations
        wf *= self.pupil #MODIFIED BY ME, WAS ABERATIONS NOW IS PUPIL

        # Propagate
        shift = sign * centre
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        wf = wf.propagate(self.spike_npixels, pixel_scale, shift=shift)

        # Return PSF
        return wf.psf

    def get_diffraction_angles(self, wavelengths):
        """
        Method to get the diffraction angles for a given wavelength set.
        """
        period = self.grating_period * 1e-6  # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2)  # Radians
        return dlu.rad2arcsec(angles)

    def model_spikes(self, wavelengths, offset, weights):
        """ """
        # Get center shift values
        period = self.grating_period * 1e-6  # Convert to meters
        angles = np.arcsin(wavelengths / period) / np.sqrt(2)  # Radians
        # angles = get_diffraction_angles(wavelengths)
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        centre = angles.mean(0) // pixel_scale

        # Model
        signs = np.array([[-1, +1], [+1, +1], [-1, -1], [+1, -1]])
        propagator = vmap(self.model_spike, (None, None, None, None, 0, None))
        return propagator(wavelengths, offset, weights, angles, signs, centre)

    def full_model(self, source, cent_nwavels=5):
        """
        Returns the diffraction spikes of the PSF

        source should be an MixedAplhaCen object
        """
        if not isinstance(source, MixedAlphaCen()):
            raise TypeError("source must be a MixedAlphaCen object")

        # Get Values
        wavelengths = source.wavelengths
        weights = source.norm_weights
        fluxes = source.raw_fluxes
        positions = source.xy_positions
        fratio = source.mixing

        # Calculate relative fluxes
        # TODO: Translate grating depth to central vs corner flux
        # Requires some experimental  mathematics
        # Probably requires both period and depth
        central_flux = 0.8
        corner_flux = 0.2

        # Model Central
        # TODO: Downsample central wavelengths and weights
        central_wavelegths = wavelengths
        central_weights = weights
        propagator = vmap(self.propagate, in_axes=(None, 0, 0))
        central_psfs = propagator(central_wavelegths, positions, central_weights)
        central_psfs *= central_flux * fluxes[:, None, None]

        # Model spikes
        propagator = vmap(self.model_spikes, in_axes=(None, 0, 0))
        spikes = propagator(wavelengths, positions, weights)
        spikes *= corner_flux * fluxes[:, None, None, None] / 4

        # Return
        return central_psfs.sum(0), spikes.sum(0)

# Adding my own class
# Have to make it its own thing. I.e. an angular optical system with a few extra parameters
# Wouldn't mind a simple toggle of struts on/off as well.
# Will take all parameters necessary for the construction of the toliman layer

# Nevermind, looks like I can just wrap it. Thanks chatgpt
# although... not sure how this will work with telescope/detector layers and dithers
# something for future me to worry about.
# I'm not sure what happened here or if I still use this code.. I think I just use the sidelobetelescope one.
class SideLobeSystem:
    def __init__(self, optics: AngularOpticalSystem):
        #self._base = base_system #stores original object
        self.optics = optics

    def __repr__(self):
        return f"SideLobeSystem(\n  optics={repr(self.optics)}\n)"
    
    def __getattr__(self, name):
        return getattr(self.optics, name)

    def compute_sidelobes(
            self, 
            grating_depth: float, 
            grating_period: float, 
            wavelength: float = None,
            corner: Array = np.array([-1,-1]),
            sources = None
            ):
        """
        A function for modelling sidelobes efficiently given an aperture.
        Note that any pupil and phase grating (i.e. abberated) layers 
        should be absent from the input optical system.

        Parameters
        ----------
        grating_depth : float
            The depth of the phase grating in meters.

        grating_period : float
            The period of the phase grating in meters.

        wavelength : float
            The wavelength to centre the sidelobe image on, in meters.

        corner : Array
            The corner to simulate. 
            [-1, -1] = bottom left, [1, -1] = bottom right, 
            [-1,  1] = top left,    [1,  1] = top right.

        sources : 
            The source object to model
        """
        # compute the sidelobes
        if isinstance(sources, PointSource):
            # calculate the offset caused by phase grating (radians)
            wavelengths = sources.wavelengths
            weights = sources.weights
            total_flux = sources.flux
            position = sources.position
            psf_pixel_scale = self.psf_pixel_scale

            # calculate the centre angle (radians)
            centre_angle = np.arcsin(wavelength/grating_period)

            pixel_scale = dlu.arcsec2rad(psf_pixel_scale)
            # make sure that the pixels are discretised correctly
            centre_angle = (centre_angle//pixel_scale)*pixel_scale

            new_center = position + corner * centre_angle

            angles = np.arcsin(wavelengths/grating_period)

            positions = np.ones((len(wavelengths),2))*new_center

            # getting fluxes of certain wavelengths

            # initialising new sources
            new_sources = [None] * len(wavelengths)

            # initialising psf
            psf_npixels = self.psf_npixels
            oversample = self.oversample
            sidelobe_psf = np.zeros((psf_npixels*oversample,psf_npixels*oversample))
            for idx, wl in enumerate(wavelengths):

                positions = positions.at[idx].set(position + corner * angles[idx] - new_center)

                new_sources[idx] = PointSource(wavelengths = np.array([wl]), 
                                               position = positions[idx], 
                                               flux = weights[idx]/np.sum(weights) * total_flux, 
                                               weights = np.array([1]))
                
                sidelobe_psf += self.model(new_sources[idx])

            # definitely ways to make the above code more efficient.
            return sidelobe_psf
        
        elif isinstance(sources, PointSources):
            return print('pee')
        
        else:
            raise TypeError(f"Unsupported source type: {type(sources)}")

# NOTE: Maybe avove parameters would be better for individual functions.

# Making the sidelobe telescope. want to time it so...
import time

# Need Bessel functions
from jax.scipy.special import j0
from jax.scipy.special import j1
# from Ben Pope:https://github.com/jax-ml/jax/pull/17038/
# I just copied and pasted the relevant parts into my jax installation

# make it an instrument
class SideLobeTelescope(dLux.Instrument):
    
    telescope: Telescope
    grating_period: float
    grating_depth: float

    # initialising
    def __init__(self, 
                 telescope: Telescope, 
                 grating_period: float, 
                 grating_depth: float):
        self.telescope = telescope
        self.grating_period = grating_period
        self.grating_depth = grating_depth

    # the reason I want this stuff here is because you can then optimise for these
    # with jax I think. i.e. can solve for the grating period
    # also could add a relative angle/offset eventually.
    # makes sense to have it here because it is part of the actual optics
    # not just an artefact of propagation
    
    # representing itself
    def __repr__(self):
        return (
            "SideLobeTelescope(\n"
            f"  telescope={repr(self.telescope)},\n"
            f"  grating_period={self.grating_period},\n"
            f"  grating_depth={self.grating_depth}\n"
            ")"
        )
    
    # getting attributes
    def __getattr__(self, name):
        return getattr(self.telescope, name)

    # define abstract method 'model', necessary for making it an instrument
    def model(self):
        return self.telescope.model()

    # define a property: sidelobe (flux) factors
    @property
    def sidelobe_factor(self):
        s_factor = (j0(self.grating_depth/4)**2) * (j1(self.grating_depth/4)**2)
        return s_factor
    
    # central factor
    @property
    def central_factor(self):
        c_factor = j0(self.grating_depth/4)**4
        return c_factor

    # now let's propagate some sidelobes.
    def model_sidelobes(
        self,
        center_wavelength: float,
        corner: Array = np.array([-1,-1]),
        center: Array = np.array([0,0]),
        assumed_pixel_scale: float = None,
        downsample: int = None
    ):
        optics = self.telescope.optics 

        # we have an 'assumed pixel scale' which calculates the position of our central offset
        # from center_wavelength
        if assumed_pixel_scale is None:
            assumed_pixel_scale = optics.psf_pixel_scale

        if isinstance(optics, AngularOpticalSystem_object):

            # calculating the central offset
            center_angle = np.arcsin(center_wavelength/self.grating_period)
            # getting pixel scale
            pixel_scale = dlu.arcsec2rad(optics.psf_pixel_scale)

            # getting the assumed pixel scale (radians)
            a_pixel_scale_rad = dlu.arcsec2rad(assumed_pixel_scale)

            # make sure that the pixels are discretised correctly
            # we divide by the assumed pixel scale with a flooring function to get the:
            # integer number of pixels shifted away.
            # we then multiply by the true pixel scale to get the true central position
            # e.g. if we assume the pixel scale is too large
            # we will have an angle smaller than necessary.
            # so the true image will be offset further away (our center will be too close)
            center_angle_corrected = np.floor(center_angle/a_pixel_scale_rad)*pixel_scale

            # new center (american spelling)
            new_center = center + corner * center_angle_corrected 

            oversample = optics.oversample
            psf_npixels = optics.psf_npixels
            # blank image
            image = np.zeros((oversample*psf_npixels, oversample*psf_npixels)) 
        else:
            print('error needs to be angular optical system')

        # calling it scene but technically can just be one source
        scene = self.source

        if not isinstance(scene, Scene):
            # forcing it to be scene (kinda jank)
            scene = Scene(scene)

        # propagating sidelobes if you have a scene:
        if isinstance(scene, Scene):
            
            # preparing the new sources
            new_sources = {}

            # using .items gives both name and values
            for name, source in scene.sources.items():

                # currently only take scenes as multiple point sources
                if isinstance(source, PointSource):
                    position = source.position
                    sidelobe_flux = source.flux * self.sidelobe_factor # to get appropriate sidelobe flux.
                    central_flux = source.flux * self.central_factor # for later
                    # not sure if need .spectrum here
                    wavelengths = source.spectrum.wavelengths
                    weights = source.spectrum.weights
                    weights_sum = np.sum(weights)

                    angles = np.arcsin(wavelengths/self.grating_period) # not paraxial, because its far from center

                    # this is really jank but going to force it into an array if it isn't currently one
                    angles = np.atleast_1d(angles)
                    wavelengths = np.atleast_1d(wavelengths)
                    
                    # make a bunch of sources, propagate once. 
                    for idx, wl in enumerate(wavelengths):
                        norm_flux = weights[idx]/weights_sum * sidelobe_flux # appropriate sidelobe flux

                        wl_position = position + corner * angles[idx] - new_center

                        mono_source = LobePointSource(
                            wavelengths = np.array([wl]),
                            position = wl_position,
                            flux = norm_flux, # to keep jax'ible..... not working... don't know why.
                            weights = np.array([1]),
                        )

                        # calling the new sources by the old sources + wavelength index
                        new_name = f"{name}_wl{idx}"
                        new_sources[new_name] = mono_source

                elif isinstance(source, AlphaCen): #careful with self/source
                    weights = source.norm_weights
                    sidelobe_fluxes = source.raw_fluxes * self.sidelobe_factor
                    central_fluxes = source.raw_fluxes * self.central_factor
                    positions = source.xy_positions
                    wavelengths = 1e-9 * source.wavelengths
                    
                    angles = np.arcsin(wavelengths/self.grating_period)

                    # iterate for each star
                    for star in range(weights.shape[0]):

                        # iterate over wavelengths
                        for idx, wl in enumerate(wavelengths):
                            norm_flux = weights[star, idx] * sidelobe_fluxes[star] # for sidelobe

                            wl_position = positions[star] + corner * angles[idx] - new_center

                            mono_source = LobePointSource(
                                wavelengths = np.array([wl]),
                                position = wl_position,
                                flux = np.array(norm_flux),
                                weights = np.array([1]),
                            )

                            # calling the source by old source + star + wavelength index
                            new_name = f"{name}_star{star}_wl{idx}"
                            new_sources[new_name] = mono_source

                else:
                    print('error: scene needs to be point sources')
            
            # Assuming you can reinitialize the telescope with all original attributes
            sidelobe_model_telescope = self.telescope.__class__(  # Create a new instance of the same class
                optics = self.telescope.optics,
                source = Scene(sources=list(new_sources.items())),
                detector = self.telescope.detector
            )
            start = time.time()
            sidelobe_image = sidelobe_model_telescope.model()
            # adding downsample parameter
            if downsample is not None:
                sidelobe_image = dlu.downsample(sidelobe_image, downsample, False)
            end = time.time()
            print(f"Model time: {end-start:.4f} seconds.")
            return sidelobe_image         
        else:
            print('error needs to be scene of point sources')
    
    def model_4_sidelobes(
        self,
        center_wavelength: float,
        center: Array = np.array([0,0]),
        assumed_pixel_scale: float = None,
        downsample: int = None
    ):

        corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
        sidelobes = [] # initialising it
        for corner in corners:
            result = self.model_sidelobes(
                center_wavelength = center_wavelength,
                corner = corner,
                center = center,
                assumed_pixel_scale = assumed_pixel_scale,
                downsample = downsample
            )

            sidelobes.append(result)

        return np.array(sidelobes)

### Making a new system for climb optimisation of the grating

from copy import deepcopy

def strip_layer(optics, layer_name="pupil"):
    # Get items in a uniform way
    if isinstance(optics.layers, dict):
        items = list(optics.layers.items())
    else:
        items = list(optics.layers)  # already list of (name, layer)

    # Build the new layers as a list of (name, layer) tuples
    new_layers = []
    for name, layer in items:
        if name == layer_name:
            continue
        # must be (str, OpticalLayer) — not strings, not names only
        new_layers.append((name, deepcopy(layer)))

    # Construct a plain AngularOpticalSystem with the pupil removed
    return AngularOpticalSystem_object(
        wf_npixels      = optics.wf_npixels,
        diameter        = optics.diameter,
        layers          = new_layers,              # <-- list of (name, OpticalLayer)
        psf_npixels     = optics.psf_npixels,
        oversample      = optics.oversample,
        psf_pixel_scale = optics.psf_pixel_scale,
    )

class SideLobeCLIMB(dLux.Instrument):
    
    telescope: Telescope
    grating_period: float
    grating_depth: float
    middle_wavelength: float
    assumed_pixel_scale: float

    # initialising
    def __init__(self, 
                 telescope: Telescope, 
                 grating_period: float, 
                 grating_depth: float,
                 middle_wavelength: float,
                 assumed_pixel_scale: float = None):
        self.telescope = telescope
        self.grating_period = grating_period
        self.grating_depth = grating_depth
        self.middle_wavelength = middle_wavelength
        self.assumed_pixel_scale = assumed_pixel_scale

    # the reason I want this stuff here is because you can then optimise for these
    # with jax I think. i.e. can solve for the grating period
    # also could add a relative angle/offset eventually.
    # makes sense to have it here because it is part of the actual optics
    # not just an artefact of propagation
    
    # representing itself
    def __repr__(self):
        return (
            "SideLobeTelescope(\n"
            f"  telescope={repr(self.telescope)},\n"
            f"  grating_period={self.grating_period},\n"
            f"  grating_depth={self.grating_depth}\n"
            ")"
        )
    
    # getting attributes
    def __getattr__(self, name):
        return getattr(self.telescope, name)

    # define a property: sidelobe (flux) factors
    @property
    def sidelobe_factor(self):
        s_factor = (j0(self.grating_depth/4)**2) * (j1(self.grating_depth/4)**2)
        return s_factor
    
    # central factor
    @property
    def central_factor(self):
        c_factor = j0(self.grating_depth/4)**4
        return c_factor
    
    # define abstract method 'model', necessary for making it an instrument
    def model(self):
        optics = self.telescope.optics

        # we have an 'assumed pixel scale' which calculates the position of our central offset
        # from center_wavelength
        assumed_pixel_scale = self.assumed_pixel_scale
        
        if assumed_pixel_scale is None:
            assumed_pixel_scale = optics.psf_pixel_scale

        center = np.array([0,0])
        corner = np.array([1,1])

        center_wavelength = self.middle_wavelength

        if isinstance(optics, AngularOpticalSystem_object):

            # calculating the central offset
            center_angle = np.arcsin(center_wavelength/self.grating_period)
            # getting pixel scale
            pixel_scale = dlu.arcsec2rad(optics.psf_pixel_scale)

            # getting the assumed pixel scale (radians)
            a_pixel_scale_rad = dlu.arcsec2rad(assumed_pixel_scale)

            # make sure that the pixels are discretised correctly
            # we divide by the assumed pixel scale with a flooring function to get the:
            # integer number of pixels shifted away.
            # we then multiply by the true pixel scale to get the true central position
            # e.g. if we assume the pixel scale is too large
            # we will have an angle smaller than necessary.
            # so the true image will be offset further away (our center will be too close)
            center_angle_corrected = np.floor(center_angle/a_pixel_scale_rad)*pixel_scale

            # new center (american spelling)
            new_center = center + corner * center_angle_corrected 

            oversample = optics.oversample
            psf_npixels = optics.psf_npixels
            # blank image
            image = np.zeros((oversample*psf_npixels, oversample*psf_npixels)) 
        else:
            print('error needs to be angular optical system')

        # calling it scene but technically can just be one source
        scene = self.source

        if not isinstance(scene, Scene):
            # forcing it to be scene (kinda jank)
            scene = Scene(scene)

        # propagating sidelobes if you have a scene:
        if isinstance(scene, Scene):
            
            # preparing the new sources
            new_sources = {}

            # making new central sources
            new_central_sources = {}

            # using .items gives both name and values
            for name, source in scene.sources.items():

                # currently only take scenes as multiple point sources
                if isinstance(source, PointSource):
                    position = source.position
                    sidelobe_flux = 4 * source.flux * self.sidelobe_factor # factor of 4 for 4 sidelobes
                    central_flux = source.flux * self.central_factor # for later
                    # not sure if need .spectrum here
                    wavelengths = source.spectrum.wavelengths
                    weights = source.spectrum.weights
                    weights_sum = np.sum(weights)

                    angles = np.arcsin(wavelengths/self.grating_period) # not paraxial, because its far from center

                    # this is really jank but going to force it into an array if it isn't currently one
                    angles = np.atleast_1d(angles)
                    wavelengths = np.atleast_1d(wavelengths)
                    
                    # make a bunch of sources, propagate once. 
                    for idx, wl in enumerate(wavelengths):
                        norm_flux = weights[idx]/weights_sum * sidelobe_flux # appropriate sidelobe flux

                        wl_position = position + corner * angles[idx] - new_center

                        mono_source = LobePointSource(
                            wavelengths = np.array([wl]),
                            position = wl_position,
                            flux = norm_flux, # to keep jax'ible..... not working... don't know why.
                            weights = np.array([1]),
                        )

                        # calling the new sources by the old sources + wavelength index
                        new_name = f"{name}_wl{idx}"
                        new_sources[new_name] = mono_source
                    
                    central_source = LobePointSource(
                        wavelengths = wavelengths,
                        position = position,
                        flux = central_flux,
                        weights = weights
                    )

                    new_central_name = f"{name}_central"
                    new_central_sources[new_central_name] = central_source

                # elif isinstance(source, AlphaCen): #careful with self/source
                #     weights = source.norm_weights
                #     sidelobe_fluxes = 4 * source.raw_fluxes * self.sidelobe_factor
                #     central_fluxes = source.raw_fluxes * self.central_factor
                #     positions = source.xy_positions
                #     wavelengths = 1e-9 * source.wavelengths
                    
                #     angles = np.arcsin(wavelengths/self.grating_period)

                #     # iterate for each star
                #     for star in range(weights.shape[0]):

                #         # iterate over wavelengths
                #         for idx, wl in enumerate(wavelengths):
                #             norm_flux = weights[star, idx] * sidelobe_fluxes[star] # for sidelobe

                #             wl_position = positions[star] + corner * angles[idx] - new_center

                #             mono_source = LobePointSource(
                #                 wavelengths = np.array([wl]),
                #                 position = wl_position,
                #                 flux = np.array(norm_flux),
                #                 weights = np.array([1]),
                #             )

                #             # calling the source by old source + star + wavelength index
                #             new_name = f"{name}_star{star}_wl{idx}"
                #             new_sources[new_name] = mono_source

                else:
                    print('error: scene needs to be point sources')
            
            # Assuming you can reinitialize the telescope with all original attributes
            sidelobe_model_telescope = self.telescope.__class__(  # Create a new instance of the same class
                optics = self.telescope.optics,
                source = Scene(sources=list(new_sources.items())),
                detector = self.telescope.detector
            )

            central_optics = strip_layer(self.telescope.optics, "pupil")
            central_model_telescope = self.telescope.__class__(
                optics = central_optics,
                source = Scene(sources=list(new_central_sources.items())),
                detector = self.telescope.detector
            )

            start = time.time()
            sidelobe_image = sidelobe_model_telescope.model()
            #central_image = central_model_telescope.model()

            end = time.time()
            print(f"Model time: {end-start:.4f} seconds.")

            # gonna ignore central optics
            return sidelobe_image
            #return np.array([central_image, sidelobe_image])
                 
        else:
            print('error needs to be scene of point sources')

        #return self.telescope.model()