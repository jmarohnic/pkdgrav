##################################################################
# ssedit.py
# Julian C. Marohnic
# Created: 7/1/22
# Modified: 1/31/23
# 
# A set of functions to aid in manipulating data from ss files.
# Includes "Particle" and "Assembly" classes, along with provision
# for units.
##################################################################

import numpy as np
import scipy.spatial as ss
import matplotlib.pyplot as plt
import sys
import ssio
from mpl_toolkits.mplot3d import Axes3D

# Particle data structure. Attributes correspond to standard pkdgrav particle parameters.
class Particle:
    def __init__(self, iOrder, iOrgIdx, m, R, x=0.0, y=0.0, z=0.0,
                 vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0, color=3,
                 units='pkd'):
        self.iOrder = iOrder
        self.iOrgIdx = iOrgIdx
        self.m = m
        self.R = R
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.color = color
        self._units = units # Need to initialize like this to avoid bugs with the units setter function.
        self.units = units

    # Input checking for attributes where this makes sense.
    @property
    def iOrder(self):
        return self._iOrder

    @iOrder.setter
    def iOrder(self, value):
        if not isinstance(value, int):
            raise TypeError("iOrder must be a non-negative integer.")
        if value < 0:
            raise ValueError("iOrder must be a non-negative integer.")
        self._iOrder = value

    @property
    def iOrgIdx(self):
        return self._iOrgIdx

    @iOrgIdx.setter
    def iOrgIdx(self, value):
        if not isinstance(value, int):
            raise TypeError("iOrgIdx must be an integer.")
        self._iOrgIdx = value

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("Particle mass must be a positive number.")
        if value <= 0:
            raise ValueError("Particle mass must be a positive number.")
        self._m = float(value)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("Particle radius must be a positive number.")
        if value <= 0:
            raise ValueError("Particle radius must be a positive number.")
        self._R = float(value)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if not isinstance(value, int):
            raise TypeError("Particle color must be an integer between 0 and 255.")
        if value < 0 or value > 255:
            raise ValueError("Particle radius must be an integer between 0 and 255.")
        self._color = value

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value not in ['pkd', 'cgs', 'mks']:
            raise ValueError("Particle units must be one of 'pkd', 'cgs', or 'mks'. Default is 'pkd'.") 
        self.convert(value)

    # Return particle position vector. Can come in handy.
    def pos(self):
        return np.array([self.x, self.y, self.z])

    # Return particle velocity vector.
    def vel(self):
        return np.array([self.vx, self.vy, self.vz])

    # Return particle spin vector.
    def spin(self):
        return np.array([self.wx, self.wy, self.wz])

    # Set particle position with a vector input.
    def set_pos(self, pos, units=None):
        if not isinstance(pos, np.ndarray) and not isinstance(pos, tuple) and not isinstance(pos, list):
            raise TypeError("Input position must be a 3-element vector.")
        if len(pos) != 3:
            raise ValueError("Input position must be a 3-element vector.")
        # If units are supplied, use them, otherwise default to particle's current units.
        if units != None:
            self.units = units

        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    # Set particle velocity with a vector input.
    def set_vel(self, vel, units=None):
        if not isinstance(vel, np.ndarray) and not isinstance(vel, tuple) and not isinstance(vel, list):
            raise TypeError("Input velocity must be a 3-element vector.")
        if len(vel) != 3:
            raise ValueError("Input velocity must be a 3-element vector.")
        if units != None:
            self.units = units

        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]

    # Set particle spin with a vector input.
    def set_spin(self, w, units=None):
        if not isinstance(w, np.ndarray) and not isinstance(w, tuple) and not isinstance(w, list):
            raise TypeError("Input spin must be a 3-element vector.")
        if len(w) != 3:
            raise ValueError("Input spin must be a 3-element vector.")
        if units != None:
            self.units = units

        self.wx = w[0]
        self.wy = w[1]
        self.wz = w[2]

    # Create a new particle with the same parameter values.
    def copy(self):
        return Particle(self.iOrder, self.iOrgIdx, self.m, self.R, self.x, self.y, self.z,
                 self.vx, self.vy, self.vz, self.wx, self.wy, self.wz, self.color, self.units)

    # Convert particle units. New particle units assumed to be 'pkd' unless specified.
    def convert(self, value='pkd'):
        if value == 'pkd':
            if self.units == 'pkd':
                return None
            elif self.units == 'cgs':
                cgs2pkd(self)
                return None
            elif self.units == 'mks':
                mks2pkd(self)
                return None
        elif value == 'cgs':
            if self.units == 'cgs':
                return None
            elif self.units == 'pkd':
                pkd2cgs(self) 
                return None
            elif self.units == 'mks':
                mks2cgs(self)
                return None
        elif value == 'mks':
            if self.units == 'mks':
                return None
            elif self.units == 'pkd':
                pkd2mks(self)
                return None
            elif self.units == 'cgs':
                cgs2mks(self)
                return None
        else:
            raise ValueError("Something has gone wrong here!")
            return 1

    # Display particle attributes when printed.
    def __str__(self):
        return (f'[{self.iOrder}, {self.iOrgIdx}, {self.m}, {self.R}, '
                f'{self.x}, {self.y}, {self.z}, {self.vx}, {self.vy}, {self.vz}, '
                f'{self.wx}, {self.wy}, {self.wz}, {self.color}, {self.units}]')


# "Assembly" data structure. Holds 1 or more particles. This could maybe be implemented in a cleaner way.
# NOTE: Must unpack list argument to Assembly(). E.g.: Assembly(*<list of particles>), NOT Assembly(<list of particles>)
class Assembly(list):
    def __init__(self, *particles, units='pkd'):
        # Establish units for the assembly, using the "property" approach below to verify input. 'pkd' is default.
        self.units = units
        # We initialize assembly with a copy of each input particle to keep the assembly indepenent from the particles
        # it's made from.
        to_add = []
        for element in particles:
            if not isinstance(element, Particle):
                raise TypeError("All assembly elements must be particles.")
            p_copy = element.copy()
            # Make sure each particle is using units consistent with the assembly choice.
            if p_copy.units != self.units:
                p_copy.convert(units)

            to_add.append(p_copy)

        list.__init__(self, to_add)

    # Manage units for the assembly. Whenever changing, check that choice is valid and ensure each particle is consistent.
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value not in ['pkd', 'cgs', 'mks']:
            raise TypeError("Assembly units must be one of 'pkd', 'cgs', or 'mks'. Default is 'pkd'.") 
        for particle in self:
            if particle.units != value:
                particle.convert(value)
        self._units = value

    def N(self):
        return len(self)

    def M(self):
        return sum([particle.m for particle in self])

    # Return a tuple with the minimum and maximum x positions over all particles in the assembly. There may be a nicer way to package this info?
    # Originally called "xrange" etc., but unfortunately this name was already taken by a Python builtin.
    def xbounds(self):
        allx = [particle.x for particle in self]
        return (min(allx), max(allx))

    def ybounds(self):
        ally = [particle.y for particle in self]
        return (min(ally), max(ally))

    def zbounds(self):
        allz = [particle.z for particle in self]
        return (min(allz), max(allz))

    # Return center of mass position of assembly.
    def com(self):
        pos = 0.0
        for particle in self:
            pos += particle.m*particle.pos()
        return pos/self.M()
    
    # Return center of mass velocity of assembly.
    def comv(self):
        vel = 0.0
        for particle in self:
            vel += particle.m*particle.vel() 
        return vel/self.M()

    # Calculate "center" of an assembly. I.e., the point halfway between the min and max x, y, and z values.
    # Not equivalent to center of mass!
    def center(self):
        xmin, xmax = self.xbounds()
        ymin, ymax = self.ybounds()
        zmin, zmax = self.zbounds()

        centx = xmin + (xmax - xmin)/2
        centy = ymin + (ymax - ymin)/2
        centz = zmin + (zmax - zmin)/2

        return np.array([centx, centy, centz])

    # Translate all particles so that the assembly has the new center of mass position.
    # Roundoff error could be an issue here when moving com by a large amount. Should 
    # be fine when aggs are relatively small?
    def set_com(self, com, units=None):
        if not isinstance(com, np.ndarray) and not isinstance(com, tuple) and not isinstance(com, list):
            raise TypeError("Input center of mass must be a 3-element vector.")
        if len(com) != 3:
            raise ValueError("Input center of mass must be a 3-element vector.")
        # If units are supplied, use them, otherwise default to assembly's current units.
        if units != None:
            self.units = units

        # Ensure that com is a numpy array and preserve original center of mass location.
        com = np.array(com)
        current_com = self.com()

        for particle in self:
            rel_pos = particle.pos() - current_com
            particle.set_pos(com + rel_pos)

    # Edit all particles so that the assembly has the new center of mass velocity.
    def set_comv(self, comv, units=None):
        if not isinstance(comv, np.ndarray) and not isinstance(comv, tuple) and not isinstance(comv, list):
            raise TypeError("Input velocity must be a 3-element vector.")
        if len(comv) != 3:
            raise ValueError("Input velocity must be a 3-element vector.")
        if units != None:
            self.units = units

        comv = np.array(comv)
        current_comv = self.comv()

        for particle in self:
            rel_vel = particle.vel() - current_comv
            particle.set_vel(comv + rel_vel)

    # Translate assembly to match the desired center location. Occasionally useful, especially 
    # when making figures. Default behavior with no argument is to move center to (0,0,0).
    def set_center(self, center=(0,0,0), units=None):
        if not isinstance(center, np.ndarray) and not isinstance(center, tuple) and not isinstance(center, list):
            raise TypeError("Input velocity must be a 3-element vector.")
        if len(center) != 3:
            raise ValueError("Input velocity must be a 3-element vector.")
        if units != None:
            self.units = units

        # Calculate necessary displacement.
        current_cent = self.center()
        disp = center - current_cent

        for particle in self:
            pos = particle.pos()
            particle.set_pos(pos + disp)

    # Get an estimate of assembly radius. Naive approach here just picks greatest distance
    # between assembly COM and any one particle in the assembly, plus that particle's radius.
    def R(self):
        if self.N() < 2:
            raise ValueError("Assembly radius calculation requires at least 2 particles.")
        COM = self.com()
        distances = [np.linalg.norm(particle.pos() - COM) + particle.R for particle in self]
        return max(distances)

    # Calculate volume occupied by assembly particles. Uses convex hull.
    # Clearly, usefullness and accuracy will depend on the inputs.
    def vol(self):
        if self.N() < 4:
            raise ValueError("Volume calculation requires at least 4 particles. See scipy.spatial.ConvexHull() for docs.")
        points = [(particle.x, particle.y, particle.z) for particle in self]
        hull = ss.ConvexHull(points)
        return hull.volume

    # Average particle density, as opposed to bulk density below. Confusing, consider removing.
    def avg_dens(self):
        dens = []
        for particle in self:
            dens.append(particle.m/((4/3)*np.pi*particle.R**3))
        return np.average(dens)

    def bulk_dens(self):
        if self.N() < 4:
            raise TypeError("Volume calculation requires at least 4 particles. See scipy.spatial.ConvexHull() for docs.")
        return self.M()/self.vol()

    # Return the inertia tensor of the assembly. Taken from pkdgrav function pkdAggsGetAxesAndSpin() in aggs.c.
    def I(self):
        com = self.com()
        comv = self.comv()

        I_matrix = np.zeros((3,3))

        for particle in self:
            # Save particle's mass, radius, moment of inertia prefactor, relative position, and relative velocity for ease of access.
            m = particle.m
            R = particle.R
            q = 0.4*R**2
            r = particle.pos() - com
            v = particle.vel() - comv

            I_matrix[0][0] += m*(q + r[1]**2 + r[2]**2)
            I_matrix[0][1] -= m*r[0]*r[1]
            I_matrix[0][2] -= m*r[0]*r[2]
            I_matrix[1][1] += m*(q + r[0]**2 + r[2]**2)
            I_matrix[1][2] -= m*r[1]*r[2]
            I_matrix[2][2] += m*(q + r[0]**2 + r[1]**2)

        I_matrix[1][0] = I_matrix[0][1]
        I_matrix[2][0] = I_matrix[0][2]
        I_matrix[2][1] = I_matrix[1][2]

        return I_matrix

    # Return the principal axes of the assembly. Just get the eigenvectors of the inertia tensor.
    def axes(self):
        I = self.I()

        return np.linalg.eig(I)[1]

    # Return the angular momentum vector of the assembly.
    def L(self):
        com = self.com()
        comv = self.comv()
        
        L_vector = np.zeros(3)

        for particle in self:
            m = particle.m
            R = particle.R
            q = 0.4*R**2
            r = particle.pos() - com
            v = particle.vel() - comv
            w = particle.spin()
            p = m*v

            L_vector += np.cross(r, p)
            L_vector += q*m*w

        return L_vector


    # Probably superfluous after adding __str__
    def show_particles(self):
        for particle in self:
            print(particle)

    # Add a copy of a particle to an assembly. Future manipulations of the assembly will not affect the original particle.
    # Not sure this is the right approach, this behavior may change in the future.
    def add_particles(self, *particles):
        for element in particles:
            if not isinstance(element, Particle):
                raise TypeError("Can only add particles to assemblies.")
            self.append(element.copy())

    # This will return a *new* particle with the same properties as the one requested. To pick out the specified particle itself
    # from the assembly to edit, user can use list slicing. E.g.: <assembly>.get_particle(7) will return a copy of the particle
    # with iOrder 7 in the agg.
    def get_particle(self, iOrder):
        for particle in self:
            if particle.iOrder == iOrder:
                return particle.copy()
        print("No particle with the given iOrder was found.")

    # When calling, need to unpack any list arguments: <assembly>.del_particles(*<list with particles to be deleted>)
    def del_particles(self, *iOrders):
        for element in iOrders:
            if not isinstance(element, int):
                raise TypeError("del_particles() can only take integers (iOrder values) as arguments.\n"
                                "If you would like to use a list to specify the particles to be deleted, use the '*' operator.\n" 
                                "E.g.: <assembly>.del_particles(*<list with particles to be deleted>)")
        # Need to do it this way. Deleting elements while iterating over a list is not good. Create new list w/ desired 
        # particles and overwrite existing assembly.
        self[:] = Assembly(*[particle for particle in self if particle.iOrder not in iOrders], units=self.units)

    def copy(self):
        return Assembly(*[particle.copy() for particle in self], units=self.units)

    # Sort assmblies by iOrder. Optional 'direction' argument allows sorting in ascending or descending order.
    def sort_iOrder(self, direction='a'):
        if direction == 'a':
            self.sort(key=iOrder_key)
        elif direction == 'd':
            self.sort(key=iOrder_key, reverse=True)
        else:
            raise ValueError("Error: direction argument can be 'a' for ascending or 'd' for descending.  Default is ascending.") 

    # Similar to the preceding function, but with opposite default behavior.
    def sort_iOrgIdx(self, direction='d'):
        if direction == 'a':
            self.sort(key=iOrgIdx_key)
        elif direction == 'd':
            self.sort(key=iOrgIdx_key, reverse=True)
        else:
            raise ValueError("Error: direction argument can be 'a' for ascending or 'd' for descending. Default is descending.")
    
    # Renumber iOrders consecutively, either ascending or descending.
    def condense(self, direction='a'):
        self.sort_iOrder()
        for i, particle in enumerate(self):
            particle.iOrder = i

        if direction == 'd':
            self.sort_iOrder('d')

    # Allows sensible printing of assembly contents.
    def __str__(self):
        to_print = ''
        for particle in self:
            to_print += (f'[{particle.iOrder}, {particle.iOrgIdx}, {particle.m:E}, {particle.R:E}, {particle.x:E}, {particle.y:E}, {particle.z:E}, '
                         f'{particle.vx:E}, {particle.vy:E}, {particle.vz:E}, {particle.wx:E}, {particle.wy:E}, {particle.wz:E}, {particle.color}, {particle.units}]\n')
        # Remove final new line character before returning.
        return to_print[:-1]

    ### AGG METHODS ###

    # Find agg with largest (negative) index.
    def agg_max(self):
        return min([particle.iOrgIdx for particle in self])

    def agg_min(self):
        return max([particle.iOrgIdx for particle in self if particle.iOrgIdx < 0])

    def agg_range(self):
        agg_tags = [particle.iOrgIdx for particle in self if particle.iOrgIdx < 0]
        return (max(agg_tags), min(agg_tags))

    def agg_list(self):
        # Determine whether the assembly contains any aggregates.
        agg_list = []
        for particle in self:
            if particle.iOrgIdx < 0:
                agg_list.append(particle.iOrgIdx)
        
        # Find unique elemnts of agg_list, print number and iOrgIdx values if any aggs exist.
        agg_list = list(set(agg_list))
        if len(agg_list) == 0:
            print("No aggs in this assembly.")
            return None
        else:
            # Call to set() above destroys order.
            agg_list.sort(reverse=True)
            return agg_list

    # Return number of aggs in the assembly.
    def N_aggs(self):
        return len(self.agg_list())

    # Returns a new assembly consisting only of particles in the desired aggregate.
    def get_agg(self, iOrgIdx):
        if not isinstance(iOrgIdx, int):
            raise TypeError("Warning: get_agg() takes a single negative integer as its argument.")
        if iOrgIdx >= 0:
            raise ValueError("Warning: get_agg() takes a single negative integer as its argument.")

        matches = [particle for particle in self if particle.iOrgIdx == iOrgIdx]
        return Assembly(*matches, units=self.units)

    # Delete specified agg from the assembly.
    def del_agg(self, iOrgIdx):
        if not isinstance(iOrgIdx, int):
            raise TypeError("Warning: del_agg() takes a single negative integer as its argument.")
        if iOrgIdx >= 0:
            raise ValueError("Warning: del_agg() takes a single negative integer as its argument.")

        del_list = [particle.iOrder for particle in self if particle.iOrgIdx == iOrgIdx]
        self.del_particles(*del_list)

    # "Pop" the desired agg from the assembly, deleting it from the assembly and returning a new copy.
    def pop_agg(self, iOrgIdx):
        if not isinstance(iOrgIdx, int):
            raise TypeError("Warning: pop_agg() takes a single negative integer as its argument.")
        if iOrgIdx >= 0:
            raise ValueError("Warning: pop_agg() takes a single negative integer as its argument.")

        del_list = [particle.iOrder for particle in self if particle.iOrgIdx == iOrgIdx]
        matches = [particle for particle in self if particle.iOrgIdx == iOrgIdx]

        new = Assembly(*matches, units=self.units)
        self.del_particles(*del_list)

        return new

    # Find any single particles with iOrgIdx < 0 ("orphans") and set iOrgIdx = iOrder.
    # Currently very slow. Consider ways to make this operation more efficient.
    def fix_orphans(self):
        agg_tags = [particle.iOrgIdx for particle in self if particle.iOrgIdx < 0]
        orphans = []
        for index in agg_tags:
            if agg_tags.count(index) == 1:
                orphans.append(index)

        if len(orphans) == 0:
            print("No orphan particles found in this assembly.")
            return None

        for particle in self:
            if particle.iOrgIdx in orphans:
                particle.iOrgIdx = particle.iOrder

        print(len(orphans), "orphan(s) corrected.")

    # Rotate entire assembly by the specified angle about the specified axis. Note that this will be a rotation
    # about the origin regardless of where the assembly is centered. Maybe add something to allow user to specify in the future.
    def rotate(self, axis, angle):
        # Kludge to prevent weird stuff from happening in these cases. If angle is 
        # zero or no legit axis is specified, do nothing.
        if np.linalg.norm(axis) == 0 or angle == 0:
            raise ValueError("Warning: invalid axis or angle of rotation passed to rotate() method. No rotation was performed.")

        for particle in self:
            rotated = vector_rotate(particle.pos(), axis, angle)

            particle.x = rotated[0]
            particle.y = rotated[1]
            particle.z = rotated[2]

# Functions

def ss_in(filename, units='pkd'):
    try:
        _, ssdata = ssio.read_SS(filename, 'y')
    except:
        print("Error: Invalid ss file.")
        return 1

    new_assembly = Assembly()
    for i in range(len(ssdata[0])):
        new_assembly.add_particles(Particle(ssdata[0,i], ssdata[1,i], ssdata[2,i], ssdata[3,i],
                                            ssdata[4,i], ssdata[5,i], ssdata[6,i], ssdata[7,i], 
                                            ssdata[8,i], ssdata[9,i], ssdata[10,i], ssdata[11,i],
                                            ssdata[12,i], ssdata[13,i]))

    # Set units if needed. Default is pkd, from Assembly().
    if units != 'pkd':
        new_assembly.units = units

    return new_assembly

def ss_out(assembly, filename):
    if not isinstance(assembly, Assembly):
        raise TypeError("Only assemblies can be written to ss files.")

    # Make sure we are writing real particles. 
    bad_list = []
    for i, element in enumerate(assembly):
        if not isinstance(element, Particle):
            bad_list.append(i)

    if bad_list != []:
        raise TypeError("Can only add particles to assemblies. The following assembly element(s) are not proper particles:\n"
                        f"{bad_list}.")

    # Warn user about any duplicate and non-sequential iOrder fields. ssio.py
    # will always renumber particles as 0,1,2,... Need to reorder to do this.
    # Don't want to reorder the actual assembly before writing, so need to make a copy. 
    ss_copy = assembly.copy()
    ss_copy.sort_iOrder()

    dup_list = []
    seq_warn = 0

    for i in range(1, ss_copy.N()):
        if ss_copy[i].iOrder == ss_copy[i-1].iOrder:
            dup_list.append(ss_copy[i].iOrder)
        if ss_copy[i].iOrder != i:
            seq_warn = 1

    if dup_list != []:
        print("Warning: the following iOrder numbers appear *at least* twice in the assembly to be written:\n" 
              f"{list(set(dup_list))}.")

    if seq_warn == 1:
        print("Warning: the iOrder values of the particles you are trying to write are not\n"
              "in sequential, increasing order beginning with '0'. This numbering will not\n"
              "be respected by ssio.py. You may call the 'condense()' method on your assembly\n"
              "to see how your particles will be renumbered.")

    # Ensure that units are 'pkd' before writing.
    ss_copy.units = 'pkd'

    # Pack up data for writing with ssio.
    iOrder_list = [particle.iOrder for particle in ss_copy]
    iOrgIdx_list = [particle.iOrgIdx for particle in ss_copy]
    m_list = [particle.m for particle in ss_copy]
    R_list = [particle.R for particle in ss_copy]
    x_list = [particle.x for particle in ss_copy]
    y_list = [particle.y for particle in ss_copy]
    z_list = [particle.z for particle in ss_copy]
    vx_list = [particle.vx for particle in ss_copy]
    vy_list = [particle.vy for particle in ss_copy]
    vz_list = [particle.vz for particle in ss_copy]
    wx_list = [particle.wx for particle in ss_copy]
    wy_list = [particle.wy for particle in ss_copy]
    wz_list = [particle.wz for particle in ss_copy]
    color_list = [particle.color for particle in ss_copy]

    new_ss = np.array([iOrder_list, iOrgIdx_list, m_list, R_list, \
                       x_list, y_list, z_list, vx_list, vy_list, vz_list, \
                       wx_list, wy_list, wz_list, color_list])

    try:
        ssio.write_SS(new_ss, filename)
    except:
        print("Error: Write to ss file failed.")
        return 1

# Returns a new assembly with *only* the input iOrders.
def subasbly(assembly, *iOrders, units='None'):
    sub = Assembly(*[particle.copy() for particle in assembly if particle.iOrder in iOrders])
    if units != 'None':
        sub.units = units
    else:
        sub.units = assembly.units

    return sub

# Key function for sorting assemblies by iOrder
def iOrder_key(particle):
    return particle.iOrder

# Key function for sorting assemblies by iOrgIdx
def iOrgIdx_key(particle):
    return particle.iOrgIdx
  


### UNIT FUNCTIONS ###

# Base conversions. l, m, and t are length, mass, and time, respectively.
l_pkd2cgs = 1.495978707e13
l_pkd2mks = 1.495978707e11
l_mks2cgs = 1.0e2
m_pkd2cgs = 1.98847e33
m_pkd2mks = 1.98847e30
m_mks2cgs = 1.0e3
t_pkd2cgs = 5.02254803e6
t_pkd2mks = 5.02254803e6

# Tedious unit conversion utilities. Could make this a little slicker, but I've opted for
# simplicity and ease of debugging. Numbers above courtesy of Wikipedia.
def pkd2cgs(particle):
    particle.m = m_pkd2cgs*particle.m
    particle.R = l_pkd2cgs*particle.R
    particle.x = l_pkd2cgs*particle.x
    particle.y = l_pkd2cgs*particle.y
    particle.z = l_pkd2cgs*particle.z
    particle.vx = (l_pkd2cgs/t_pkd2cgs)*particle.vx
    particle.vy = (l_pkd2cgs/t_pkd2cgs)*particle.vy
    particle.vz = (l_pkd2cgs/t_pkd2cgs)*particle.vz
    particle.wx = (1.0/t_pkd2cgs)*particle.wx
    particle.wy = (1.0/t_pkd2cgs)*particle.wy
    particle.wz = (1.0/t_pkd2cgs)*particle.wz
    particle._units = 'cgs'

def cgs2pkd(particle):
    particle.m = (1.0/m_pkd2cgs)*particle.m
    particle.R = (1.0/l_pkd2cgs)*particle.R
    particle.x = (1.0/l_pkd2cgs)*particle.x
    particle.y = (1.0/l_pkd2cgs)*particle.y
    particle.z = (1.0/l_pkd2cgs)*particle.z
    particle.vx = (t_pkd2cgs/l_pkd2cgs)*particle.vx
    particle.vy = (t_pkd2cgs/l_pkd2cgs)*particle.vy
    particle.vz = (t_pkd2cgs/l_pkd2cgs)*particle.vz
    particle.wx = t_pkd2cgs*particle.wx
    particle.wy = t_pkd2cgs*particle.wy
    particle.wz = t_pkd2cgs*particle.wz
    particle._units = 'pkd' 

def pkd2mks(particle):
    particle.m = m_pkd2mks*particle.m 
    particle.R = l_pkd2mks*particle.R
    particle.x = l_pkd2mks*particle.x
    particle.y = l_pkd2mks*particle.y
    particle.z = l_pkd2mks*particle.z
    particle.vx = (l_pkd2mks/t_pkd2mks)*particle.vx
    particle.vy = (l_pkd2mks/t_pkd2mks)*particle.vy
    particle.vz = (l_pkd2mks/t_pkd2mks)*particle.vz
    particle.wx = (1.0/t_pkd2mks)*particle.wx
    particle.wy = (1.0/t_pkd2mks)*particle.wy
    particle.wz = (1.0/t_pkd2mks)*particle.wz
    particle._units = 'mks'

def mks2pkd(particle):
    particle.m = (1.0/m_pkd2mks)*particle.m 
    particle.R = (1.0/l_pkd2mks)*particle.R
    particle.x = (1.0/l_pkd2mks)*particle.x
    particle.y = (1.0/l_pkd2mks)*particle.y
    particle.z = (1.0/l_pkd2mks)*particle.z
    particle.vx = (t_pkd2mks/l_pkd2mks)*particle.vx
    particle.vy = (t_pkd2mks/l_pkd2mks)*particle.vy
    particle.vz = (t_pkd2mks/l_pkd2mks)*particle.vz
    particle.wx = t_pkd2mks*particle.wx
    particle.wy = t_pkd2mks*particle.wy
    particle.wz = t_pkd2mks*particle.wz
    particle._units = 'pkd' 

def mks2cgs(particle):
    particle.m = m_mks2cgs*particle.m
    particle.R = l_mks2cgs*particle.R
    particle.x = l_mks2cgs*particle.x
    particle.y = l_mks2cgs*particle.y
    particle.z = l_mks2cgs*particle.z
    particle.vx = l_mks2cgs*particle.vx
    particle.vy = l_mks2cgs*particle.vy
    particle.vz = l_mks2cgs*particle.vz
    particle._units = 'cgs'

def cgs2mks(particle):
    particle.m = (1.0/m_mks2cgs)*particle.m
    particle.R = (1.0/l_mks2cgs)*particle.R
    particle.x = (1.0/l_mks2cgs)*particle.x
    particle.y = (1.0/l_mks2cgs)*particle.y
    particle.z = (1.0/l_mks2cgs)*particle.z
    particle.vx = (1.0/l_mks2cgs)*particle.vx
    particle.vy = (1.0/l_mks2cgs)*particle.vy
    particle.vz = (1.0/l_mks2cgs)*particle.vz
    particle._units = 'mks'









#### !!!EXPERIMENTAL FEATURES!!! ####
# Beyond this point lies a mess of cool stuff that may or may not work!
# Some important notes are hidden in the comments below, I should really 
# collect all this info...





# Embed a spherical boulder in a rubble pile. Center argument is defined
# relative to agg COM.
def embed_boulder(assembly, center, radius, units='pkd'):
    COM = assembly.com()
    agg_max = min([particle.iOrgIdx for particle in assembly]) # Most negative iOrgIdx is last agg.
    boulder_list = []
    for particle in assembly:
        if np.linalg.norm(particle.pos() - (COM + center)) <= radius:
            boulder_list.append(particle.iOrgIdx)

    for particle in assembly:
        if particle.iOrgIdx in boulder_list:
            particle.iOrgIdx = agg_max - 1
            particle.color = 1

# General rotation of a vector, expressed in terms of axis and angle. Copied from ssgen2Agg.py/Wikipedia.
# We'll use this for specifying the orientation when we generate aggs. Investigate the merits of a
# quaternion approach?
def vector_rotate(vector, axis, angle):
    sa = np.sin(angle)
    ca = np.cos(angle)
    dot = np.dot(vector, axis)
    x = vector[0]
    y = vector[1]
    z = vector[2]
    u = axis[0]
    v = axis[1]
    w = axis[2]

    rotated = np.zeros(3)

    rotated[0] = u*dot+(x*(v*v+w*w) - u*(v*y+w*z))*ca + (-w*y+v*z)*sa
    rotated[1] = v*dot+(y*(u*u+w*w) - v*(u*x+w*z))*ca + (w*x-u*z)*sa
    rotated[2] = w*dot+(z*(u*u+v*v) - w*(u*x+v*y))*ca + (-v*x+u*y)*sa

    return rotated

# Calculate the angle between two vectors. Used when setting orientation of generated aggs. Copied from a post on Stack Overflow:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
# Generate an assembly consisting of a single dumbbell-shaped aggregate. User may specify a mass and "radius" for the whole agg, 
# or for each particle (using the pmass and pradius arguments in lieu of mass and radius). Suggest passing keyword arguments only to avoid confusion. 
# Specifying orientation in this way obviously leaves some degeneracy in the final attitude of the agg but gives the user some degree of
# control. This may be improved in the future.
def make_db(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=2, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    # Ensure that user supplies at least one set of mass/radius arguments.
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    # Make sure user doesn't overconstrain the agg.
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    # Avoiding overlap between 'units' as an argument of this function and 'units' as an Assembly attribute.
    temp_units = units

    # If user has specified agg mass or radius, set *particle* mass and radius (pmass and pradius) accordingly and continue.
    # Aggregate "radius" is taken to be the maximum distance between agg center and a particle edge. The "separation" s (or "sep")
    # is given by sep_coeff*pradius and is taken to be the distance between particle centers in the dumbbell case.
    # Need to account for this when converting between agg radius and particle radius.
    if mass > 0 or radius > 0: 
        pmass = mass/2
        pradius = radius*(1 + sep_coeff/2)**(-1)

    # Set particle separation for placement within the agg and set relative center positions. These will be translated
    # by the specified agg center location after the desired orientation is applied.
    sep = sep_coeff*pradius
    p0_center = np.array([0,0,-sep/2])
    p1_center = np.array([0,0,+sep/2])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, units=temp_units)

    # Set orientation. (0,0,1) is the default. Again, note that particle centers are translated *after* agg is rotated.
    # Order of operations matters here! Check first if rotation is necessary to avoid annoying warnings from rotate() method.
    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        # Normalize axis.
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg

# Generate a planar diamond-shaped aggregate.    
def make_diamond(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=12, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    # In the case of planar diamonds, the relation between particle radius and agg radius is different. Agg radius is s + pradius.
    if mass > 0 or radius > 0: 
        pmass = mass/4
        pradius = radius*(1 + sep_coeff)**(-1)

    sep = sep_coeff*pradius
    p0_center = np.array([0,0,-sep])
    p1_center = np.array([0,0,+sep])
    p2_center = np.array([-sep/2,0,0])
    p3_center = np.array([+sep/2,0,0])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg

# Generate a tetrahedron-shaped aggregate. This function generates a tetrahedron with one flat face on the bottom and an upright
# pyramid-like orientation. This is in contrast with genTetrahedron() in ssgen2Agg.py, which uses a much more elegant formulation
# for a tetrahedron with 2 level edges, but lacking an upright orientation. Since we care about orientation here, we use the former
# approach. Default position should have centroid at the origin.
def make_tetra(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=3, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    if mass > 0 or radius > 0: 
        pmass = mass/4
        pradius = radius*(1 + sep_coeff/2)**(-1)

    # Set particle separation for placement within the agg and set relative center positions. These will be translated
    # by the specified agg center location after the desired orientation is applied. Given the weird coordinates needed for
    # laying out the tetrahedron, we specify the locations of vertices on the unit circle as vectors and scale by sep/2.
    sep = sep_coeff*pradius
    p0_center = (sep/2)*np.array([np.sqrt(8/9),0,-1/3])
    p1_center = (sep/2)*np.array([-np.sqrt(2/9),np.sqrt(2/3),-1/3])
    p2_center = (sep/2)*np.array([-np.sqrt(2/9),-np.sqrt(2/3),-1/3])
    p3_center = (sep/2)*np.array([0,0,1])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg


# Generate a 4-particle rod-shaped aggregate. Default orientation is along the z-axis.
def make_rod(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=5, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    # Placement of particles is straightforward. agg radius is 1.5*sep + pradius.
    if mass > 0 or radius > 0: 
        pmass = mass/4
        pradius = radius*(1 + 3*sep_coeff/2)**(-1)

    sep = sep_coeff*pradius
    p0_center = np.array([0,0,-3*sep/2])
    p1_center = np.array([0,0,-sep/2])
    p2_center = np.array([0,0,+sep/2])
    p3_center = np.array([0,0,+3*sep/2])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg

# Generate an 8-particle cube. Centered at the origin and faces parallel to x-, y-, and z- axes.
def make_cube(iOrder=0, iOrgIdx=-1, mass=0, radius=0, center=(0,0,0), orientation=(0,0,1), color=7, pmass=0, pradius=0, sep_coeff=np.sqrt(3), units='pkd'):
    if (mass <= 0 or radius <= 0) and (pmass <= 0 or pradius <= 0):
        raise ValueError("One pair of either mass and radius, or pmass and pradius must both be positive.")
    if (radius > 0 and pradius > 0) or (mass > 0 and pmass > 0):
        raise ValueError("Specify *either* aggregate mass and radius, *or* mass and radius of consituent particles.")

    temp_units = units

    # agg radius is sqrt(3)*sep/2 + pradius.
    if mass > 0 or radius > 0: 
        pmass = mass/8
        pradius = radius*(1 + np.sqrt(3)*sep_coeff/2)**(-1)

    sep = sep_coeff*pradius
    p0_center = np.array([+sep/2,+sep/2,-sep/2])
    p1_center = np.array([+sep/2,-sep/2,-sep/2])
    p2_center = np.array([-sep/2,-sep/2,-sep/2])
    p3_center = np.array([-sep/2,+sep/2,-sep/2])
    p4_center = np.array([+sep/2,+sep/2,+sep/2])
    p5_center = np.array([+sep/2,-sep/2,+sep/2])
    p6_center = np.array([-sep/2,-sep/2,+sep/2])
    p7_center = np.array([-sep/2,+sep/2,+sep/2])
    p0 = Particle(iOrder, iOrgIdx, pmass, pradius, p0_center[0], p0_center[1], p0_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p1 = Particle(iOrder + 1, iOrgIdx, pmass, pradius, p1_center[0], p1_center[1], p1_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p2 = Particle(iOrder + 2, iOrgIdx, pmass, pradius, p2_center[0], p2_center[1], p2_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p3 = Particle(iOrder + 3, iOrgIdx, pmass, pradius, p3_center[0], p3_center[1], p3_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p4 = Particle(iOrder + 4, iOrgIdx, pmass, pradius, p4_center[0], p4_center[1], p4_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p5 = Particle(iOrder + 5, iOrgIdx, pmass, pradius, p5_center[0], p5_center[1], p5_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p6 = Particle(iOrder + 6, iOrgIdx, pmass, pradius, p6_center[0], p6_center[1], p6_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    p7 = Particle(iOrder + 7, iOrgIdx, pmass, pradius, p7_center[0], p7_center[1], p7_center[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color, units=temp_units)
    
    agg = Assembly(p0, p1, p2, p3, p4, p5, p6, p7, units=temp_units)

    axis = np.cross(np.array([0,0,1]), np.array(orientation))
    angle = angle_between(np.array([0,0,1]), np.array(orientation))

    if np.linalg.norm(axis) == 0 or angle == 0:
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg
    else:
        axis = axis/np.linalg.norm(axis)
        agg.rotate(axis, angle)
        for particle in agg:
            particle.x += center[0]
            particle.y += center[1]
            particle.z += center[2]
        return agg



# Return the coordinates for plotting a sphere centered at (x,y,z)
# Taken from Stack Overflow post: https://stackoverflow.com/questions/70977042/how-to-plot-spheres-in-3d-with-plotly-or-another-library
def makesphere(x, y, z, radius, resolution=10):
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


# Visualize an assembly of particles on the fly. Resolution determines how round or blocky the 
# particles appear. Higher res takes longer to render. Large numbers of particles won't work well here.
# Default value is 10, so manually setting a higher value will take longer to render and vice versa
# when setting a lower value.
def viz(assembly, resolution=10):
    if not isinstance(assembly, Assembly):
        raise TypeError("Only assemblies can be visualized.")
    if assembly.N() < 2:
        raise ValueError("Only assemblies with 2 or more particles may be visualized.")

    xlist = []
    ylist = []
    zlist = []
    radlist = []
    colorlist = []

    for particle in assembly:
        xlist.append(particle.x)
        ylist.append(particle.y)
        zlist.append(particle.z)
        radlist.append(particle.R)
        colorlist.append(color_translate(particle.color))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Hack to get square projection since matplotlib somehow does not support this currently. Find the center point of the assembly
    # between the most extreme particles in x, y, and z. This should be the origin of the plot. Will cause problems when there are particles
    # separated by large distances, but this should be fine for now.
    centx = assembly.xbounds()[0] + (assembly.xbounds()[1] - assembly.xbounds()[0])/2
    centy = assembly.ybounds()[0] + (assembly.ybounds()[1] - assembly.ybounds()[0])/2
    centz = assembly.zbounds()[0] + (assembly.zbounds()[1] - assembly.zbounds()[0])/2
    cent = [centx, centy, centz]
    # Would be nice to scale each axis individually, but this will lead to weirdly stretched
    # particles. matplotlib currently does not support setting a 'square' aspect ratio in 3D.
    max_dist = assembly.R()
    max_rad = max(radlist)
    
    for x, y, z, rad, color in zip(xlist, ylist, zlist, radlist, colorlist):
        X, Y, Z = makesphere(x, y, z, rad, resolution=resolution)
        ax.plot_surface(X, Y, Z, color=color)

    ax.axes.set_xlim3d(left=cent[0]-max_dist-max_rad, right=cent[0]+max_dist+max_rad)
    ax.axes.set_ylim3d(bottom=cent[1]-max_dist-max_rad, top=cent[1]+max_dist+max_rad)
    ax.axes.set_zlim3d(bottom=cent[2]-max_dist-max_rad, top=cent[2]+max_dist+max_rad)
    ax.set_xlabel(f'x ({assembly.units})')
    ax.set_ylabel(f'y ({assembly.units})')
    ax.set_zlabel(f'z ({assembly.units})')

    plt.show()

# Single-use utility for converting pkd colors to matplotlib colors in viz().
def color_translate(pkd_color):
    if pkd_color == 0:
        return 'black'
    elif pkd_color == 1:
        return 'white'
    elif pkd_color == 2:
        return 'red'
    elif pkd_color == 3:
        return 'lawngreen'
    elif pkd_color == 4:
        return 'blue'
    elif pkd_color == 5:
        return 'yellow'
    elif pkd_color == 6:
        return 'magenta'
    elif pkd_color == 7:
        return 'cyan'
    elif pkd_color == 8:
        return 'gold'
    elif pkd_color == 9:
        return 'pink'
    elif pkd_color == 10:
        return 'orange'
    elif pkd_color == 11:
        return 'khaki'
    elif pkd_color == 12:
        return 'mediumpurple'
    elif pkd_color == 13:
        return 'maroon'
    elif pkd_color == 14:
        return 'aqua'
    elif pkd_color == 15:
        return 'navy'
    elif pkd_color == 16:
        return 'black'
    else:
        return 'gray'

# Combine existing assemblies into one new assembly. User may supply units, otherwise new assembly 
# will default to pkd units. Similar to other functions/methods for manipulating assemblies, join()
# will create a new assembly composed of copies of the input assemblies, and further manipulations
# will not affect the original inputs.
def join(*assemblies, units='pkd'):
    # Check for valid specification of units.
    if units not in ['pkd', 'mks', 'cgs']:
        raise ValueError("Valid units arguments are 'pkd', 'mks', and 'cgs'.")
    for element in assemblies:
        if not isinstance(element, Assembly):
            raise ValueError("Can only join assemblies.")

    new = Assembly()

    for element in assemblies:
        new.add_particles(*element)

    # Make sure all added particles conform to specified units before returning.
    new.units = units
    return new
