# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/11/7 21:13
from __future__ import annotations

from typing import Tuple, Any, Sequence, Callable, Optional

import torch
import torch.nn.functional as F

from .random import (identity_matrix, identity_quaternions, identity_translatins,
                     random_rotations, random_quaternions, random_translations)
from .rotation_conversions import (rotation_6d_to_matrix, quaternion_to_matrix, matrix_to_quaternion,
                                   quaternion_multiply, quaternion_invert, euler_angles_to_matrix)
from .so3 import so3_scale_matrix, so3_lerp_matrix


def rot_vec_mul(rot, vec):
    vec = vec.unsqueeze(-1)
    return (rot @ vec).squeeze(-1)


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


class Rotation:
    """
        A 3D rotation. Depending on how the object is initialized, the
        rotation is represented by either a rotation matrix or a
        quaternion, though both formats are made available by helper functions.
        To simplify gradient computation, the underlying format of the
        rotation cannot be changed in-place. Like Rigid, the class is designed
        to mimic the behavior of a torch Tensor, almost as if each Rotation
        object were a tensor of rotations, in one format or another.
    """

    def __init__(self,
                 rot_mats: Optional[torch.Tensor] = None,
                 quats: Optional[torch.Tensor] = None,
                 normalize_quats: bool = True
                 ):
        """
        Args:
            rot_mats:
                A [*, 3, 3] rotation matrix tensor. Mutually exclusive with
                quats
            quats:
                A [*, 4] quaternion. Mutually exclusive with rot_mats. If
                normalize_quats is not True, must be a unit quaternion
            normalize_quats:
                If quats is specified, whether to normalize quats
        """
        if ((rot_mats is None and quats is None) or
                (rot_mats is not None and quats is not None)):
            raise ValueError("Exactly one input argument must be specified")

        if ((rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or
                (quats is not None and quats.shape[-1] != 4)):
            raise ValueError(
                "Incorrectly shaped rotation matrix or quaternion"
            )

        # Force full-precision
        if quats is not None:
            quats = quats.to(dtype=torch.float32)

        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)

        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
            shape,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: bool = True,
            fmt: str = "quat",
    ) -> Rotation:
        """
            Returns an identity Rotation.

            Args:
                shape:
                    The "shape" of the resulting Rotation object. See documentation
                    for the shape property
                dtype:
                    The torch dtype for the rotation
                device:
                    The torch device for the new rotation
                requires_grad:
                    Whether the underlying tensors in the new rotation object
                    should require gradient computation
                fmt:
                    One of "quat" or "rot_mat". Determines the underlying format
                    of the new object's rotation
            Returns:
                A new identity rotation
        """
        # TODO: add ortho6d format
        if fmt in ("rot_mat", "rot"):
            rot_mats = identity_matrix(
                shape, dtype, device, requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt in ("quat", "quaternion"):
            quats = identity_quaternions(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    @staticmethod
    def random(
            shape,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: bool = True,
            fmt: str = "quat",
    ) -> Rotation:
        """
            Returns an identity Rotation.

            Args:
                shape:
                    The "shape" of the resulting Rotation object. See documentation
                    for the shape property
                dtype:
                    The torch dtype for the rotation
                device:
                    The torch device for the new rotation
                requires_grad:
                    Whether the underlying tensors in the new rotation object
                    should require gradient computation
                fmt:
                    One of "quat" or "rot_mat". Determines the underlying format
                    of the new object's rotation
            Returns:
                A new identity rotation
        """
        if fmt in ("rot_mat", "rot"):
            rot_mats = random_rotations(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt in ("quat", "quaternion"):
            quats = random_quaternions(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    # Magic methods
    def __getitem__(self, index: Any) -> Rotation:
        """
            Allows torch-style indexing over the virtual shape of the rotation
            object. See documentation for the shape property.

            Args:
                index:
                    A torch index. E.g. (1, 3, 2), or (slice(None,))
            Returns:
                The indexed rotation
        """
        if not isinstance(index, torch.Tensor):
            if type(index) != tuple:
                index = (index,)

            index = index + (slice(None), slice(None))
            if self._rot_mats is None:
                index = index[:-1]

        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index]
            return Rotation(rot_mats=rot_mats)
        elif self._quats is not None:
            quats = self._quats[index]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __setitem__(self, index: Any, value: Rotation) -> Rotation:
        """
            Allows torch-style indexing over the virtual shape of the rotation
            object. See documentation for the shape property.

            Args:
                index:
                    A torch index. E.g. (1, 3, 2), or (slice(None,))
            Returns:
                The indexed rotation
        """
        if not isinstance(index, torch.Tensor):
            if type(index) != tuple:
                index = (index,)

            index = index + (slice(None), slice(None))
            if self._rot_mats is not None:
                index = index[:-1]

        if self._rot_mats is not None:
            self._rot_mats[index] = value._rot_mats
            return Rotation(rot_mats=self._rot_mats)
        elif self._quats is not None:
            self._quats[index] = value._quats
            return Rotation(quats=self._quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __matmul__(self, right: Rotation) -> Rotation:
        return self.compose(right)

    def __mul__(self,
                right: torch.Tensor,
                ) -> Rotation:
        """
        Pointwise left multiplication of the rotation with a tensor. Can be
        used to e.g. mask the Rotation.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        if (self._rot_mats is not None):
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        elif (self._quats is not None):
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __rmul__(self,
                 left: torch.Tensor,
                 ) -> Rotation:
        """
        Reverse pointwise multiplication of the rotation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
            Returns the virtual shape of the rotation object. This shape is
            defined as the batch dimensions of the underlying rotation matrix
            or quaternion. If the Rotation was initialized with a [10, 3, 3]
            rotation matrix tensor, for example, the resulting shape would be
            [10].

            Returns:
                The virtual shape of the rotation object
        """
        if (self._quats is not None):
            s = self._quats.shape[:-1]
        else:
            s = self._rot_mats.shape[:-2]

        return s

    @property
    def dtype(self) -> torch.dtype:
        """
            Returns the dtype of the underlying rotation.

            Returns:
                The dtype of the underlying rotation
        """
        if (self._rot_mats is not None):
            return self._rot_mats.dtype
        elif (self._quats is not None):
            return self._quats.dtype
        else:
            raise ValueError("Both rotations are None")

    @property
    def device(self) -> torch.device:
        """
            The device of the underlying rotation

            Returns:
                The device of the underlying rotation
        """
        if (self._rot_mats is not None):
            return self._rot_mats.device
        elif (self._quats is not None):
            return self._quats.device
        else:
            raise ValueError("Both rotations are None")

    @property
    def requires_grad(self) -> bool:
        """
            Returns the requires_grad property of the underlying rotation

            Returns:
                The requires_grad property of the underlying tensor
        """
        if (self._rot_mats is not None):
            return self._rot_mats.requires_grad
        elif (self._quats is not None):
            return self._quats.requires_grad
        else:
            raise ValueError("Both rotations are None")

    def get_rot_mats(self) -> torch.Tensor:
        """
            Returns the underlying rotation as a rotation matrix tensor.

            Returns:
                The rotation as a rotation matrix tensor
        """
        rot_mats = self._rot_mats
        if rot_mats is None:
            if self._quats is None:
                raise ValueError("Both rotations are None")
            else:
                rot_mats = quaternion_to_matrix(self._quats)

        return rot_mats

    def get_quats(self) -> torch.Tensor:
        """
            Returns the underlying rotation as a quaternion tensor.

            Depending on whether the Rotation was initialized with a
            quaternion, this function may call torch.linalg.eigh.

            Returns:
                The rotation as a quaternion tensor.
        """
        quats = self._quats
        if quats is None:
            if self._rot_mats is None:
                raise ValueError("Both rotations are None")
            else:
                quats = matrix_to_quaternion(self._rot_mats)

        return quats

    def get_cur_rot(self) -> torch.Tensor:
        """
            Return the underlying rotation in its current form

            Returns:
                The stored rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError("Both rotations are None")

    def compose(self, r: Rotation):
        if self._quats is not None and r._quats is not None:
            return self.compose_q(r)
        else:
            return self.compose_r(r)

    def compose_r(self, r: Rotation) -> Rotation:
        """
        Compose the rotation matrices of the current Rotation object with
        those of another.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        new_rot_mats = r1 @ r2  # torch.matmul
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: Rotation, normalize_quats: bool = True) -> Rotation:
        """
            Compose the quaternions of the current Rotation object with those
            of another.

            Depending on whether either Rotation was initialized with
            quaternions, this function may call torch.linalg.eigh.

            Args:
                r:
                    An update rotation object
            Returns:
                An updated rotation object
        """
        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = quaternion_multiply(q1, q2)
        return Rotation(
            rot_mats=None, quats=new_quats, normalize_quats=normalize_quats
        )

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply the current Rotation as a rotation matrix to a set of 3D
        coordinates.

        Args:
            pts:
                A [*, 3] set of points

        Returns:
            [*, 3] rotated points
        """
        rot_mats = self.get_rot_mats()
        dtype = pts.dtype
        is_half = dtype in (torch.float16, torch.bfloat16)
        if is_half:
            pts = pts.float()

        pts = rot_vec_mul(rot_mats, pts)
        if is_half:
            pts = pts.to(dtype)

        return pts

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
            The inverse of the apply() method.

            Args:
                pts:
                    A [*, 3] set of points
            Returns:
                [*, 3] inverse-rotated points
        """
        dtype = pts.dtype
        is_half = dtype in (torch.float16, torch.bfloat16)
        if is_half:
            pts = pts.float()

        rot_mats = self.get_rot_mats()
        inv_rot_mats = invert_rot_mat(rot_mats)
        pts = rot_vec_mul(inv_rot_mats, pts)

        if is_half:
            pts = pts.to(dtype)

        return pts

    def invert(self) -> Rotation:
        """
        Returns the inverse of the current Rotation.

        Returns:
            The inverse of the current Rotation
        """
        if self._rot_mats is not None:
            return Rotation(
                rot_mats=invert_rot_mat(self._rot_mats),
                quats=None
            )
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=quaternion_invert(self._quats),
                normalize_quats=True,
            )
        else:
            raise ValueError("Both rotations are None")

    def scale(self, scalars: torch.Tensor) -> Rotation:
        """Scale the magnitude of a rotation matrix,
        e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.
        This is the same as taking matrix powers, but pytorch only supports integer exponents

        So instead, we take advantage of the properties of rotation matrices
        to calculate logarithms easily. and multiply instead.
        """
        out = so3_scale_matrix(self.get_rot_mats(), scalars)
        return Rotation(out)

    def lerp(self, rot: Rotation, weights: torch.Tensor) -> Rotation:
        out = so3_lerp_matrix(self.get_rot_mats(), rot.get_rot_mats(), weights)
        return Rotation(out)

    def unsqueeze(self,
                  dim: int,
                  ) -> Rotation:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the
        shape of the Rotation object.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed Rotation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        if (self._rot_mats is not None):
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif (self._quats is not None):
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    @staticmethod
    def cat(
            rs: Sequence[Rotation],
            dim: int,
    ) -> Rotation:
        """
            Concatenates rotations along one of the batch dimensions. Analogous
            to torch.cat().

            Note that the output of this operation is always a rotation matrix,
            regardless of the format of input rotations.

            Args:
                rs:
                    A list of rotation objects
                dim:
                    The dimension along which the rotations should be
                    concatenated
            Returns:
                A concatenated Rotation object in rotation matrix format
        """
        rot_mats = [r.get_rot_mats() for r in rs]
        rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(rot_mats=rot_mats, quats=None)

    @staticmethod
    def stack(
            rs: Sequence[Rotation],
            dim: int,
    ) -> Rotation:
        """
            Concatenates rotations along one of the batch dimensions. Analogous
            to torch.cat().

            Note that the output of this operation is always a rotation matrix,
            regardless of the format of input rotations.

            Args:
                rs:
                    A list of rotation objects
                dim:
                    The dimension along which the rotations should be
                    concatenated
            Returns:
                A concatenated Rotation object in rotation matrix format
        """
        rot_mats = [r.get_rot_mats() for r in rs]
        rot_mats = torch.stack(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(rot_mats=rot_mats, quats=None)

    def map_tensor_fn(self,
                      fn: Callable[torch.Tensor, torch.Tensor]
                      ) -> Rotation:
        """
            Apply a Tensor -> Tensor function to underlying rotation tensors,
            mapping over the rotation dimension(s). Can be used e.g. to sum out
            a one-hot batch dimension.

            Args:
                fn:
                    A Tensor -> Tensor function to be mapped over the Rotation
            Returns:
                The transformed Rotation object
        """
        if (self._rot_mats is not None):
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(
                list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1
            )
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif (self._quats is not None):
            quats = torch.stack(
                list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1
            )
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def cuda(self) -> Rotation:
        """
        Analogous to the cuda() method of torch Tensors

        Returns:
            A copy of the Rotation in CUDA memory
        """
        if (self._rot_mats is not None):
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        elif (self._quats is not None):
            return Rotation(
                rot_mats=None,
                quats=self._quats.cuda(),
                normalize_quats=False
            )
        else:
            raise ValueError("Both rotations are None")

    def to(self,
           device: Optional[torch.device],
           dtype: Optional[torch.dtype]
           ) -> Rotation:
        """
        Analogous to the to() method of torch Tensors

        Args:
            device:
                A torch device
            dtype:
                A torch dtype
        Returns:
            A copy of the Rotation using the new device and dtype
        """
        if (self._rot_mats is not None):
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype),
                quats=None,
            )
        elif (self._quats is not None):
            return Rotation(
                rot_mats=None,
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    def detach(self) -> Rotation:
        """
            Returns a copy of the Rotation whose underlying Tensor has been
            detached from its torch graph.

            Returns:
                A copy of the Rotation whose underlying Tensor has been detached
                from its torch graph
        """
        if (self._rot_mats is not None):
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        elif (self._quats is not None):
            return Rotation(
                rot_mats=None,
                quats=self._quats.detach(),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    def clone(self) -> Rotation:
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.clone(), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.clone(),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")


class Rigid:
    """
        A class representing a rigid transformation. Little more than a wrapper
        around two objects: a Rotation object and a [*, 3] translation
        Designed to behave approximately like a single torch tensor with the 
        shape of the shared batch dimensions of its component parts.
    """

    def __init__(self,
                 rots: Optional[Rotation],
                 trans: Optional[torch.Tensor],
                 ):
        """
        Args:
            rots: A [*, 3, 3] rotation tensor
            trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if rots is None:
            rots = Rotation.identity(
                batch_dims, dtype, device, requires_grad,
            )
        elif trans is None:
            trans = identity_translatins(
                batch_dims, dtype, device, requires_grad,
            )

        if ((rots.shape != trans.shape[:-1]) or
                (rots.device != trans.device)):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.to(dtype=torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
            shape: Tuple[int, ...],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: bool = False,
            fmt: str = "quat",
    ) -> Rigid:
        """
        Constructs an identity transformation.

        Args:
            shape:
                The desired shape
            dtype:
                The dtype of both internal tensors
            device:
                The device of both internal tensors
            requires_grad:
                Whether grad should be enabled for the internal tensors
        Returns:
            The identity transformation
        """
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_translatins(shape, dtype, device, requires_grad),
        )

    @staticmethod
    def random(
            shape: Tuple[int, ...],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: bool = False,
            fmt: str = "quat",
    ) -> Rigid:
        """
        Constructs an identity transformation.

        Args:
            shape:
                The desired shape
            dtype:
                The dtype of both internal tensors
            device:
                The device of both internal tensors
            requires_grad:
                Whether grad should be enabled for the internal tensors
        Returns:
            The identity transformation
        """
        return Rigid(
            Rotation.random(shape, dtype, device, requires_grad, fmt=fmt),
            random_translations(shape, dtype, device, requires_grad),
        )

    def __getitem__(self,
                    index: Any,
                    ) -> Rigid:
        """ 
        Indexes the affine transformation with PyTorch-style indices.
        The index is applied to the shared dimensions of both the rotation
        and the translation.

        E.g.::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
            t = Rigid(r, torch.rand(10, 10, 3))
            indexed = t[3, 4:6]
            assert(indexed.shape == (2,))
            assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        Args:
            index: A standard torch tensor index. E.g. 8, (10, None, 3),
            or (3, slice(0, 1, None))
        Returns:
            The indexed tensor
        """
        if isinstance(index, torch.Tensor):
            index = index
            trans_index = index
        elif type(index) != tuple:
            index = (index,)
            trans_index = index + (slice(None),)
        else:
            index = index
            trans_index = index + (slice(None),)

        return Rigid(
            self._rots[index],
            self._trans[trans_index],
        )

    def __setitem__(self,
                    index: Any,
                    value: Rigid
                    ) -> Rigid:
        """
        Indexes the affine transformation with PyTorch-style indices.
        The index is applied to the shared dimensions of both the rotation
        and the translation.

        E.g.::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
            t = Rigid(r, torch.rand(10, 10, 3))
            indexed = t[3, 4:6]
            assert(indexed.shape == (2,))
            assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        Args:
            index: A standard torch tensor index. E.g. 8, (10, None, 3),
            or (3, slice(0, 1, None))
        Returns:
            The indexed tensor
        """
        if isinstance(index, torch.Tensor):
            assert len(index.shape) <= len(self._trans.shape) - 1
            index = index
            trans_index = index
        elif type(index) != tuple:
            index = (index,)
            trans_index = index + (slice(None),)
        else:
            index = index
            trans_index = index + (slice(None),)

        self._rots[index] = value.get_rot_obj()
        self._trans[trans_index] = value.get_trans()

        return Rigid(self._rots, self._trans)

    def __matmul__(self, right: Rigid) -> Rigid:
        return self.compose(right)

    def __mul__(self,
                right: torch.Tensor,
                ) -> Rigid:
        """
        Pointwise left multiplication of the transformation with a tensor.
        Can be used to e.g. mask the Rigid.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self,
                 left: torch.Tensor,
                 ) -> Rigid:
        """
        Reverse pointwise multiplication of the transformation with a
        tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    def dim(self):
        return len(self.shape)

    @property
    def shape(self) -> torch.Size:
        """
            Returns the shape of the shared dimensions of the rotation and
            the translation.
            
            Returns:
                The shape of the transformation
        """
        s = self._trans.shape[:-1]
        return s

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the device on which the Rigid's tensors are located.

        Returns:
            The device on which the Rigid's tensors are located
        """
        return self._trans.dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the Rigid's tensors are located.

        Returns:
            The device on which the Rigid's tensors are located
        """
        return self._trans.device

    @property
    def requires_grad(self) -> bool:
        return self._rots.requires_grad and self._trans.requires_grad

    def get_rot_obj(self):
        return self._rots

    def get_rots(self) -> torch.Tensor:
        """
            Getter for the rotation.

            Returns:
                The rotation object
        """
        return self._rots.get_rot_mats()

    def get_quats(self) -> torch.Tensor:
        return self._rots.get_quats()

    def get_trans(self) -> torch.Tensor:
        """
        Getter for the translation.

        Returns:
            The stored translation
        """
        return self._trans

    def compose(self,
                r: Rigid,
                ) -> Rigid:
        """
        Composes the current rigid object with another.
        T = T1 @ T2 = (R1, t1) @ (R2, t2) = (R1 @ R2, R1 @ t2 + t1)

        Args:
            r: Another Rigid object
        Returns:
            The composition of the two transformations
        """
        new_rot = self._rots.compose(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(self,
              pts: torch.Tensor,
              ) -> torch.Tensor:
        """
            Applies the transformation to a coordinate tensor.

            Args:
                pts: A [*, 3] coordinate tensor.
            Returns:
                The transformed points.
        """
        dtype = pts.dtype
        is_half = dtype in (torch.float16, torch.bfloat16)
        if is_half:
            pts = pts.float()

        pts = self._rots.apply(pts) + self._trans

        if is_half:
            pts = pts.to(dtype)

        return pts

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
            Applies the inverse of the transformation to a coordinate tensor.

            Args:
                pts: A [*, 3] coordinate tensor
            Returns:
                The transformed points.
        """
        dtype = pts.dtype
        is_half = dtype in (torch.float16, torch.bfloat16)
        if is_half:
            pts = pts.float()

        pts = pts - self._trans
        pts = self._rots.invert_apply(pts)

        if is_half:
            pts = pts.to(dtype)

        return pts

    def invert(self) -> Rigid:
        """
        Inverts the transformation.

        Returns:
            The inverse transformation.
        """
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self,
                      fn: Callable[torch.Tensor, torch.Tensor]
                      ) -> Rigid:
        """
        Apply a Tensor -> Tensor function to underlying translation and
        rotation tensors, mapping over the translation/rotation dimensions
        respectively.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rigid
        Returns:
            The transformed Rigid object
        """
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(
            list(map(fn, torch.unbind(self._trans, dim=-1))),
            dim=-1
        )

        return Rigid(new_rots, new_trans)

    def to_tensor(self, dof):
        if dof in (3, 4):
            return self.to_tensor_7()
        else:
            return self.to_tensor_4x4()

    def to_tensor_4x4(self) -> torch.Tensor:
        """
            Converts a transformation to a homogenous transformation tensor.

            Returns:
                A [*, 4, 4] homogenous transformation tensor
        """
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    def to_tensor_7(self) -> torch.Tensor:
        """
            Converts a transformation to a tensor with 7 final columns, four
            for the quaternion followed by three for the translation.

            Returns:
                A [*, 7] tensor representation of the transformation
        """
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_4x4(
            t: torch.Tensor
    ) -> Rigid:
        """
            Constructs a transformation from a homogenous transformation
            tensor.

            Args:
                t: [*, 4, 4] homogenous transformation tensor
            Returns:
                T object with shape [*]
        """
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]

        return Rigid(rots, trans)

    @staticmethod
    def from_euler(euler_angles: torch.Tensor, translations: torch.Tensor):
        rot = euler_angles_to_matrix(*torch.unbind(euler_angles, dim=-1))
        return Rigid(Rotation(rot_mats=rot, quats=None), translations)

    @staticmethod
    def from_tensor_9(
            ortho6d: torch.Tensor,
            trans: torch.Tensor,
    ) -> Rigid:
        """
        Args:
            ortho6d: [*, seq_len, 6]
            ortho6d: [*, seq_len, 3]
        """
        assert ortho6d.shape[-1] == 6 and trans.shape[-1] == 3
        rot_mats = rotation_6d_to_matrix(ortho6d)
        rots = Rotation(rot_mats=rot_mats, quats=None)

        return Rigid(rots, trans)

    @staticmethod
    def from_tensor_7(
            t: torch.Tensor,
            normalize_quats: bool = False,
    ) -> Rigid:
        if t.shape[-1] != 7:
            raise ValueError("Incorrectly shaped input tensor")

        quats, trans = t[..., :4], t[..., 4:]

        rots = Rotation(
            rot_mats=None,
            quats=quats,
            normalize_quats=normalize_quats
        )

        return Rigid(rots, trans)

    @staticmethod
    def from_tensor(t: torch.Tensor, **kwargs):
        dof = t.shape[-1] - 3
        if dof == 4:
            return Rigid.from_tensor_7(t, **kwargs)
        elif dof == 6:
            return Rigid.from_tensor_9(t[...:6], t[..., 6:9])
        else:
            return Rigid.from_tensor_4x4(t)

    @classmethod
    def get_rotations_frames(cls, positions, eps=1e-12):
        """
        Returns a local rotation frame defined by N, CA, C positions.

        Args:
            positions: [*, seq_len, 3, 3], coordinates, tensor of shape
            where the third dimension is in order of N, CA, C

        Returns:
           [*, seq_len, 3, 3], Local relative rotation frames
        """
        x1, x2, x3 = torch.unbind(positions[..., :3, :], dim=-2)
        v1 = x3 - x2
        v2 = x1 - x2
        e1 = F.normalize(v1, dim=-1, eps=eps)
        u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
        e2 = F.normalize(u2, dim=-1, eps=eps)
        e3 = torch.cross(e1, e2, dim=-1)
        R = torch.stack([e1, e2, e3], dim=-2)
        return R

    @classmethod
    def from_3_points(cls,
                      p_neg_x_axis: torch.Tensor,
                      origin: torch.Tensor,
                      p_xy_plane: torch.Tensor,
                      eps=1e-8) -> Rigid:
        """
        Constructs transformations from sets of 3 points using the Gram-Schmidt algorithm.

        Refs:
        1. Jumper et al., Highly accurate protein structure prediction with AlphaFold. Nature, 2021.
            - Supplementary information, Section 1.8.1, Algorithm 21.

        Args:
            p_neg_x_axis: [*, 3] coordinates
            origin: [*, 3] coordinates used as frame origins
            p_xy_plane: [*, 3] coordinates
            eps: Small epsilon value

        Returns:
            A transformation object of shape [*]
        """
        v2 = p_xy_plane - origin
        e1 = F.normalize((origin - p_neg_x_axis), dim=-1, eps=eps)
        u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
        e2 = F.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R = torch.stack([e1, e2, e3], dim=-1)

        rot_obj = Rotation(rot_mats=R, quats=None)

        return Rigid(rot_obj, origin)

    @classmethod
    def from_atom3_positions(cls,
                             positions,
                             mask=None,
                             eps=1e-12):
        """Constructs transformations from sets of 3 atom points [N, CA, O)]
        Args:
            positions: [*, seq_len, 3, 3]
            mask: residue mask [*, seq_len, 3]

        Returns:
            [*, seq_len] rigids
        """
        rotation = cls.get_rotations_frames(positions[..., :3, :], eps=eps)
        translation = positions[..., 1, :]

        if mask is not None:
            rotation[mask] = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
            translation[mask] = torch.zeros(3, dtype=rotation.dtype, device=rotation.device)

        rot_obj = Rotation(rot_mats=rotation, quats=None)
        return Rigid(rot_obj, translation)

    def unsqueeze(self,
                  dim: int,
                  ) -> Rigid:
        """Analogous to torch.unsqueeze. The dimension is relative to the
        shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    @staticmethod
    def cat(
            ts: Sequence[Rigid],
            dim: int,
    ) -> Rigid:
        """
        Concatenates transformations along a new dimension.

        Args:
            ts:
                A list of T objects
            dim:
                The dimension along which the transformations should be
                concatenated
        Returns:
            A concatenated transformation object
        """
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat(
            [t._trans for t in ts], dim=dim if dim >= 0 else dim - 1
        )

        return Rigid(rots, trans)

    @staticmethod
    def stack(
            ts: Sequence[Rigid],
            dim: int,
    ) -> Rigid:
        """
        Concatenates transformations along a new dimension.

        Args:
            ts:
                A list of T objects
            dim:
                The dimension along which the transformations should be
                concatenated
        Returns:
            A concatenated transformation object
        """
        rots = Rotation.stack([t._rots for t in ts], dim)
        trans = torch.stack(
            [t._trans for t in ts], dim=dim if dim >= 0 else dim - 1
        )

        return Rigid(rots, trans)

    def apply_rot_fn(self, fn: Callable[Rotation, Rotation]) -> Rigid:
        """
        Applies a Rotation -> Rotation function to the stored rotation
        object.

        Args:
            fn: A function of type Rotation -> Rotation
        Returns:
            A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn: Callable[torch.Tensor, torch.Tensor]) -> Rigid:
        """
            Applies a Tensor -> Tensor function to the stored translation.

            Args:
                fn:
                    A function of type Tensor -> Tensor to be applied to the
                    translation
            Returns:
                A transformation object with a transformed translation.
        """
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> Rigid:
        """
            Scales the translation by a constant factor.

            Args:
                trans_scale_factor:
                    The constant factor
            Returns:
                A transformation object with a scaled translation.
        """
        fn = lambda t: t * trans_scale_factor
        return self.apply_trans_fn(fn)

    def se3_scale(self, scalars: torch.Tensor):
        """Scale the magnitude of a rotation matrix,
        e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.
        This is the same as taking matrix powers, but pytorch only supports integer exponents
        Args:
            scalars: [*batch_dims]
        """
        rot = self._rots.scale(scalars)
        trans = self._trans * scalars[..., None]
        return Rigid(rot, trans)

    def se3_lerp(self, transform: Rigid, weight: torch.Tensor):
        """Weighted interpolation between transform1 and transform2"""
        rot_lerps = self._rots.lerp(transform._rots, weight)
        shift_lerps = torch.lerp(self._trans, transform._trans, weight)
        return Rigid(rot_lerps, shift_lerps)

    def stop_rot_gradient(self) -> Rigid:
        """
            Detaches the underlying rotation object

            Returns:
                A transformation object with detached rotations
        """
        fn = lambda r: r.detach()
        return self.apply_rot_fn(fn)

    def detach(self) -> Rigid:
        return Rigid(self._rots.detach(),
                     self._trans.detach())

    def clone(self) -> Rigid:
        return Rigid(self._rots.clone(),
                     self._trans.clone())

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-12):
        """
        Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you
        provide the atom positions in the non-standard way, the N atom will
        end up not at [-0.527250, 1.359329, 0.0] but instead at
        [-0.527250, -1.359329, 0.0]. You need to take care of such cases in
        your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen(N) xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha(CA) xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz(C) coordinates.

        Returns:
            A transformation object. After applying the translation and
            rotation to the reference backbone, the coordinates will
            approximately equal to the input coordinates.
        """
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = c2_rots @ c1_rots
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = n_rots @ c_rots

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self) -> Rigid:
        """
        Moves the transformation object to GPU memory

        Returns:
            A version of the transformation on GPU
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())

    def reshape(self, shape):
        trans = self.get_trans().reshape(tuple(shape) + (3,))
        quats = self.get_rot_obj()._quats
        rots = self.get_rot_obj()._rot_mats
        if quats is None:
            rots = rots.reshape(tuple(shape) + (3, 3))
        else:
            quats = quats.reshape(tuple(shape) + (4,))
        return Rigid(Rotation(rot_mats=rots, quats=quats), trans=trans)

