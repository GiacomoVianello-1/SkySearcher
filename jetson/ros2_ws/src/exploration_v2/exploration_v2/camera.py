import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog

class Camera:
    def __init__(self, h, w, fx, fy, cx, cy, depth: float = 3.0):
        self.h = h
        self.w = w
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth = depth

        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)
        
        # Rays in Camera Space
        pixel_coords = np.array([
            [0, 0, 1],      # top-left
            [w, 0, 1],      # top-right
            [0, h, 1],      # bottom-left
            [w, h, 1],      # bottom-right
        ])
        self.fov_rays = pixel_coords @ self.K_inv.T
        
        # Pre-compute local planes [A, B, C, D]
        # Normals must point OUTWARD so that internal points satisfy Ax + By + Cz + D <= 0
        tl, tr, bl, br = self.fov_rays
        origin = np.zeros(3)

        self.local_planes = np.array([
            self._get_plane(origin, tr, tl),           # Top
            self._get_plane(origin, bl, br),           # Bottom
            self._get_plane(origin, tl, bl),           # Left
            self._get_plane(origin, br, tr),           # Right
            self._get_plane(tl*depth, tr*depth, br*depth) # Far (Base)
        ])

    @staticmethod
    def _get_plane(p1, p2, p3):
        """Calculates plane equation [A, B, C, D] where Ax+By+Cz+D=0."""
        n = np.cross(p2 - p1, p3 - p1)
        n /= (np.linalg.norm(n) + 1e-12)
        d = -np.dot(n, p1)
        return np.append(n, d)

    def _transform_planes(self, pose):
        """Transforms local planes into world space using the camera pose."""
        R = pose[:3, :3]
        t = pose[:3, 3]
        world_planes = []
        for plane in self.local_planes:
            n_world = R @ plane[:3]
            # d_world = d_local - (n_world · translation)
            d_world = plane[3] - np.dot(n_world, t)
            world_planes.append(np.append(n_world, d_world))
        return np.array(world_planes)

    def in_fov(self, pose, points_world):
        """Check if world points are inside the frustum."""
        R_t = pose[:3, :3].T
        t = pose[:3, 3]
        points_local = (points_world - t) @ R_t.T
        points_pixel_hom = points_local @ self.K.T
        z = points_pixel_hom[:, 2]
        valid_z = (z > 0) & (z <= self.depth)
        u = points_pixel_hom[:, 0] / (z + 1e-6)
        v = points_pixel_hom[:, 1] / (z + 1e-6)
        return (u >= 0) & (u <= self.w) & (v >= 0) & (v <= self.h) & valid_z
    
    def project_pixel_to_ground(self, u, v, pose, h_ground):
        """
        Project a single pixel (u, v) onto the world ground plane z = h_ground
        by intersecting the corresponding camera ray with the plane.

        Returns
        -------
        point : np.ndarray of shape (3,) or None
            World-frame point on the ground plane, or None if the ray does
            not intersect the plane in front of the camera.
        slant : float or None
            Distance from the camera centre to the ground point (used to
            recover physical scale from pixel measurements).
        """
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Ray in camera frame, then transformed to world frame
        d_cam = self.K_inv @ np.array([u, v, 1.0])
        d_world = R @ d_cam

        dz = d_world[2]
        if abs(dz) < 1e-6:
            return None, None  # ray parallel to ground

        s = (h_ground - t[2]) / dz
        if s <= 0:
            return None, None  # intersection is behind the camera

        point = t + s * d_world
        slant = s * np.linalg.norm(d_world)
        return point, slant

    def intersect(self, cam_b, pose_a, pose_b):
        """
        Calculates the intersection of two camera frustums.
        Returns the vertices of the intersection and the total volume.
        """
        # 1. Generate world-space planes for both cameras
        planes_a = self._transform_planes(pose_a)
        planes_b = cam_b._transform_planes(pose_b)
        all_planes = np.vstack((planes_a, planes_b))

        # 2. Find a feasible interior point using Linear Programming
        # We need a point that satisfies Ax + By + Cz + D < 0 for all planes
        res = linprog(
            c=[0, 0, 0], 
            A_ub=all_planes[:, :3], 
            b_ub=-all_planes[:, 3] - 1e-7, # Subtract epsilon to ensure strictly inside
            bounds=(None, None),
            method='highs'
        )

        # Logical Check: If no feasible point exists, there is no intersection volume
        if not res.success:
            print("Error: No intersection volume found (Frustums do not overlap).")
            return None, 0.0

        feasible_point = res.x

        # 3. Compute the half-space intersection
        # If the intersection is degenerate (zero volume), this will raise a QhullError
        hs = HalfspaceIntersection(all_planes, feasible_point)
        vertices = hs.intersections
        
        # 4. Compute Volume
        volume = ConvexHull(vertices).volume
        
        return vertices, volume
    
    def batch_intersect(self, cams, pose_a, poses):
        """Intersect cam_a with multiple cam_b's in a batch."""
        results = []
        for cam_b, pose_b in zip(cams, poses):
            vertices, volume = self.intersect(cam_b, pose_a, pose_b)
            results.append((vertices, volume))
        return results
    
    def frustum_volume(self):
        """Calculate the volume of the camera frustum."""
        tl, tr, bl, br = self.fov_rays
        near_corners = np.array([tl, tr, bl, br])
        far_corners = near_corners * self.depth
        vertices = np.vstack((np.zeros((1, 3)), near_corners, far_corners))
        return ConvexHull(vertices).volume
    
    def plot_frustum(self, ax, pose, color='b', alpha=0.2, label=None):
        """Helper to visualize the camera pyramid in 3D."""
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Calculate world-space vertices
        # Origin + 4 corner rays scaled by depth
        origin = t
        corners = (self.fov_rays * self.depth) @ R.T + t
        
        # Define the 5 faces of the pyramid
        faces = [
            [origin, corners[0], corners[1]], # Top
            [origin, corners[2], corners[3]], # Bottom
            [origin, corners[0], corners[2]], # Left
            [origin, corners[1], corners[3]], # Right
            [corners[0], corners[1], corners[3], corners[2]] # Base (Far)
        ]
        
        poly = Poly3DCollection(faces, facecolors=color, alpha=alpha, edgecolors=color)
        ax.add_collection3d(poly)
        
        # Plot label point (at origin)
        ax.scatter(t[0], t[1], t[2], color=color, s=50, label=label)

if __name__ == "__main__":
    camera = Camera(h=720, w=1280, fx=525.0, fy=525.0, cx=640.0, cy=360.0)

    # Identity pose (camera at origin, looking along +Z)
    pose1 = np.eye(4)

    # Second camera: translated 2 units on X and rotated 30 deg on Y
    angle = np.radians(30)
    pose2 = np.array([
        [np.cos(angle),  0, np.sin(angle), 2.0],
        [0,              1, 0,             0.0],
        [-np.sin(angle), 0, np.cos(angle), 0.0],
        [0,              0, 0,             1.0],
    ])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot both cameras
    camera.plot_frustum(ax, pose1, color="b", alpha=0.1, label="Camera 1")
    camera.plot_frustum(ax, pose2, color="r", alpha=0.1, label="Camera 2")

    # Correct unpacking: intersect returns (vertices, volume)
    start = time.time()
    intersect_points, volume = camera.intersect(camera, pose1, pose2)
    print(f"Intersection computed in {time.time() - start:.4f} seconds.")
    if intersect_points is not None:
        volume_a = camera.frustum_volume()
        print(f"Intersection Volume: {volume:.4f} / {volume_a:.4f} ({volume/volume_a*100:.2f}%)")
        
        # Plot the intersection vertices
        ax.scatter(intersect_points[:, 0], intersect_points[:, 1], intersect_points[:, 2], 
                   color="g", s=30, label="Intersection Vertices")
        
        # To see the intersection shape better, draw its hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(intersect_points)
        for simplex in hull.simplices:
            ax.plot(intersect_points[simplex, 0], 
                    intersect_points[simplex, 1], 
                    intersect_points[simplex, 2], "g-", alpha=0.5)
    else:
        print("No intersection found.")

    ax.set_xlim(-1, 5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 6)
    ax.set_xlabel("X (World)")
    ax.set_ylabel("Y (World)")
    ax.set_zlabel("Z (World)")    
    ax.legend()
    plt.show()