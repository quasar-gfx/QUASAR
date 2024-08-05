cmake_minimum_required(VERSION 3.19)
project(assimp)

set(OUR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(TARGET              assimp)
set(SRC_DIR             ${OUR_DIR})
set(PUBLIC_HDR_DIR      ${OUR_DIR}/include)
set(PRIVATE_HDR_DIR     ${OUR_DIR}/code)

# public headers dependencies
set(PUBLIC_HDRS
    ${PUBLIC_HDR_DIR}/${TARGET}/AssertHandler.h
    ${PUBLIC_HDR_DIR}/${TARGET}/Base64.hpp
    ${PUBLIC_HDR_DIR}/${TARGET}/DefaultIOStream.h
    ${PUBLIC_HDR_DIR}/${TARGET}/DefaultIOSystem.h
    ${PUBLIC_HDR_DIR}/${TARGET}/IOSystem.hpp
    ${PUBLIC_HDR_DIR}/${TARGET}/SceneCombiner.h
    ${PUBLIC_HDR_DIR}/${TARGET}/ai_assert.h
    ${PUBLIC_HDR_DIR}/${TARGET}/anim.h
    ${PUBLIC_HDR_DIR}/${TARGET}/camera.h
    ${PUBLIC_HDR_DIR}/${TARGET}/cexport.h
    ${PUBLIC_HDR_DIR}/${TARGET}/cfileio.h
    ${PUBLIC_HDR_DIR}/${TARGET}/cimport.h
    ${PUBLIC_HDR_DIR}/${TARGET}/color4.h
    ${PUBLIC_HDR_DIR}/${TARGET}/config.h
    ${PUBLIC_HDR_DIR}/${TARGET}/defs.h
    ${PUBLIC_HDR_DIR}/${TARGET}/Exceptional.h
    ${PUBLIC_HDR_DIR}/${TARGET}/importerdesc.h
    ${PUBLIC_HDR_DIR}/${TARGET}/light.h
    ${PUBLIC_HDR_DIR}/${TARGET}/material.h
    ${PUBLIC_HDR_DIR}/${TARGET}/matrix3x3.h
    ${PUBLIC_HDR_DIR}/${TARGET}/matrix4x4.h
    ${PUBLIC_HDR_DIR}/${TARGET}/mesh.h
    ${PUBLIC_HDR_DIR}/${TARGET}/metadata.h
    ${PUBLIC_HDR_DIR}/${TARGET}/postprocess.h
    ${PUBLIC_HDR_DIR}/${TARGET}/quaternion.h
    ${PUBLIC_HDR_DIR}/${TARGET}/scene.h
    ${PUBLIC_HDR_DIR}/${TARGET}/texture.h
    ${PUBLIC_HDR_DIR}/${TARGET}/types.h
    ${PUBLIC_HDR_DIR}/${TARGET}/vector2.h
    ${PUBLIC_HDR_DIR}/${TARGET}/vector3.h
    ${PUBLIC_HDR_DIR}/${TARGET}/version.h
)
if (ANDROID)
set(PUBLIC_HDRS ${PUBLIC_HDRS}
    ${PUBLIC_HDR_DIR}/${TARGET}/port/AndroidJNI/AndroidJNIIOSystem.h
    ${PUBLIC_HDR_DIR}/${TARGET}/port/AndroidJNI/BundledAssetIOSystem.h
)
endif()

# private headers dependencies
set(PRIVATE_HDRS
    ${PRIVATE_HDR_DIR}/Common/BaseProcess.h
    ${PRIVATE_HDR_DIR}/Common/Compression.h
    ${PRIVATE_HDR_DIR}/CApi/CInterfaceIOWrapper.h
    ${PRIVATE_HDR_DIR}/PostProcessing/ArmaturePopulate.h
    ${PRIVATE_HDR_DIR}/PostProcessing/CalcTangentsProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/ComputeUVMappingProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/ConvertToLHProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/DeboneProcess.h
    ${PRIVATE_HDR_DIR}/Common/DefaultProgressHandler.h
    ${PRIVATE_HDR_DIR}/PostProcessing/DropFaceNormalsProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/EmbedTexturesProcess.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXCommon.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXCompileConfig.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXConverter.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXDocument.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXDocumentUtil.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXImportSettings.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXImporter.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXMeshGeometry.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXParser.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXProperties.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXTokenizer.h
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXUtil.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFCommon.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFAsset.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFImporter.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF2/glTF2Asset.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF2/glTF2Importer.h
    ${PRIVATE_HDR_DIR}/Common/FileLogStream.h
    ${PRIVATE_HDR_DIR}/Common/FileSystemFilter.h
    ${PRIVATE_HDR_DIR}/PostProcessing/FindDegenerates.h
    ${PRIVATE_HDR_DIR}/PostProcessing/FindInstancesProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/FindInvalidDataProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/FixNormalsStep.h
    ${PRIVATE_HDR_DIR}/PostProcessing/GenFaceNormalsProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/GenVertexNormalsProcess.h
    ${PRIVATE_HDR_DIR}/Common/IFF.h
    ${PRIVATE_HDR_DIR}/Common/Importer.h
    ${PRIVATE_HDR_DIR}/PostProcessing/ImproveCacheLocality.h
    ${PRIVATE_HDR_DIR}/PostProcessing/JoinVerticesProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/LimitBoneWeightsProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/MakeVerboseFormat.h
    ${PRIVATE_HDR_DIR}/Material/MaterialSystem.h
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjFileData.h
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjFileImporter.h
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjFileMtlImporter.h
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjFileParser.h
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjTools.h
    ${PRIVATE_HDR_DIR}/PostProcessing/OptimizeGraph.h
    ${PRIVATE_HDR_DIR}/PostProcessing/OptimizeMeshes.h
    ${PRIVATE_HDR_DIR}/Common/PolyTools.h
    ${PRIVATE_HDR_DIR}/PostProcessing/PretransformVertices.h
    ${PRIVATE_HDR_DIR}/PostProcessing/ProcessHelper.h
    ${PRIVATE_HDR_DIR}/PostProcessing/RemoveRedundantMaterials.h
    ${PRIVATE_HDR_DIR}/PostProcessing/RemoveVCProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/ScaleProcess.h
    ${PRIVATE_HDR_DIR}/Common/ScenePreprocessor.h
    ${PRIVATE_HDR_DIR}/Common/ScenePrivate.h
    ${PRIVATE_HDR_DIR}/PostProcessing/SortByPTypeProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/SplitByBoneCountProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/SplitLargeMeshes.h
    ${PRIVATE_HDR_DIR}/Common/StdOStreamLogStream.h
    ${PRIVATE_HDR_DIR}/Common/TargetAnimation.h
    ${PRIVATE_HDR_DIR}/PostProcessing/TextureTransform.h
    ${PRIVATE_HDR_DIR}/PostProcessing/TriangulateProcess.h
    ${PRIVATE_HDR_DIR}/PostProcessing/ValidateDataStructure.h
    ${PRIVATE_HDR_DIR}/Common/VertexTriangleAdjacency.h
    ${PRIVATE_HDR_DIR}/Geometry/GeometryUtils.h
    ${PRIVATE_HDR_DIR}/Common/Win32DebugLogStream.h
    ${PRIVATE_HDR_DIR}/Common/assbin_chunks.h
)

# sources dependencies
set(SRCS
    ${PRIVATE_HDR_DIR}/Common/Base64.cpp
    ${PRIVATE_HDR_DIR}/Common/material.cpp
    ${PRIVATE_HDR_DIR}/Common/AssertHandler.cpp
    ${PRIVATE_HDR_DIR}/Common/Assimp.cpp
    ${PRIVATE_HDR_DIR}/Common/BaseImporter.cpp
    ${PRIVATE_HDR_DIR}/Common/BaseProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/Compression.cpp
    ${PRIVATE_HDR_DIR}/Common/Bitmap.cpp
    ${PRIVATE_HDR_DIR}/CApi/CInterfaceIOWrapper.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/CalcTangentsProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/ComputeUVMappingProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/ConvertToLHProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/CreateAnimMesh.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/DeboneProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/DefaultIOStream.cpp
    ${PRIVATE_HDR_DIR}/Common/DefaultIOSystem.cpp
    ${PRIVATE_HDR_DIR}/Common/IOSystem.cpp
    ${PRIVATE_HDR_DIR}/Common/DefaultLogger.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/DropFaceNormalsProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/EmbedTexturesProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/Exceptional.cpp
    ${PRIVATE_HDR_DIR}/Common/Exporter.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXAnimation.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXBinaryTokenizer.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXConverter.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXDeformer.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXDocument.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXDocumentUtil.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXImporter.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXMaterial.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXMeshGeometry.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXModel.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXNodeAttribute.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXParser.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXProperties.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXTokenizer.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/FBX/FBXUtil.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFCommon.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFCommon.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFAsset.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFAsset.inl
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFImporter.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/glTF/glTFImporter.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF2/glTF2Asset.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF2/glTF2Asset.inl
    ${PRIVATE_HDR_DIR}/AssetLib/glTF2/glTF2Importer.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/glTF2/glTF2Importer.h
    ${PRIVATE_HDR_DIR}/AssetLib/glTF2/glTF2Importer.cpp
    # ${PRIVATE_HDR_DIR}/AssetLib/X3D/FIReader.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/ArmaturePopulate.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/FindDegenerates.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/FindInstancesProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/FindInvalidDataProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/FixNormalsStep.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/GenFaceNormalsProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/GenBoundingBoxesProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/GenVertexNormalsProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/Importer.cpp
    ${PRIVATE_HDR_DIR}/Common/ImporterRegistry.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/ImproveCacheLocality.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/JoinVerticesProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/LimitBoneWeightsProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/MakeVerboseFormat.cpp
    ${PRIVATE_HDR_DIR}/Material/MaterialSystem.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjFileImporter.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjFileMtlImporter.cpp
    ${PRIVATE_HDR_DIR}/AssetLib/Obj/ObjFileParser.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/OptimizeGraph.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/OptimizeMeshes.cpp
    ${PRIVATE_HDR_DIR}/Common/PostStepRegistry.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/PretransformVertices.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/ProcessHelper.cpp
    ${PRIVATE_HDR_DIR}/Common/RemoveComments.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/RemoveRedundantMaterials.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/RemoveVCProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/SGSpatialSort.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/ScaleProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/SceneCombiner.cpp
    ${PRIVATE_HDR_DIR}/Common/ScenePreprocessor.cpp
    ${PRIVATE_HDR_DIR}/Common/SkeletonMeshBuilder.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/SortByPTypeProcess.cpp
    ${PRIVATE_HDR_DIR}/Common/SpatialSort.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/SplitByBoneCountProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/SplitLargeMeshes.cpp
    ${PRIVATE_HDR_DIR}/Common/StandardShapes.cpp
    ${PRIVATE_HDR_DIR}/Common/Subdivision.cpp
    ${PRIVATE_HDR_DIR}/Common/TargetAnimation.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/TextureTransform.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/TriangulateProcess.cpp
    ${PRIVATE_HDR_DIR}/PostProcessing/ValidateDataStructure.cpp
    ${PRIVATE_HDR_DIR}/Common/Version.cpp
    ${PRIVATE_HDR_DIR}/Common/VertexTriangleAdjacency.cpp
    ${PRIVATE_HDR_DIR}/Common/scene.cpp
    ${PRIVATE_HDR_DIR}/Geometry/GeometryUtils.cpp
    ${SRC_DIR}/contrib/clipper/clipper.cpp
    ${SRC_DIR}/contrib/poly2tri/poly2tri/common/shapes.cc
    ${SRC_DIR}/contrib/poly2tri/poly2tri/sweep/advancing_front.cc
    ${SRC_DIR}/contrib/poly2tri/poly2tri/sweep/cdt.cc
    ${SRC_DIR}/contrib/poly2tri/poly2tri/sweep/sweep_context.cc
    ${SRC_DIR}/contrib/poly2tri/poly2tri/sweep/sweep.cc
    ${SRC_DIR}/contrib/unzip/ioapi.c
    ${SRC_DIR}/contrib/unzip/unzip.c
)
if (ANDROID)
set(SRCS ${SRCS}
    ${SRC_DIR}/port/AndroidJNI/AndroidJNIIOSystem.cpp
    ${SRC_DIR}/port/AndroidJNI/BundledAssetIOSystem.cpp
)
endif()

add_definitions(
    -DASSIMP_BUILD_NO_3D_IMPORTER
    -DASSIMP_BUILD_NO_3DS_IMPORTER
    -DASSIMP_BUILD_NO_3MF_IMPORTER
    -DASSIMP_BUILD_NO_AC_IMPORTER
    -DASSIMP_BUILD_NO_AMF_IMPORTER
    -DASSIMP_BUILD_NO_ASSBIN_IMPORTER
    -DASSIMP_BUILD_NO_ASE_IMPORTER
    -DASSIMP_BUILD_NO_B3D_IMPORTER
    -DASSIMP_BUILD_NO_BLEND_IMPORTER
    -DASSIMP_BUILD_NO_BVH_IMPORTER
    -DASSIMP_BUILD_NO_C4D_IMPORTER
    -DASSIMP_BUILD_NO_COB_IMPORTER
    -DASSIMP_BUILD_NO_COLLADA_IMPORTER
    -DASSIMP_BUILD_NO_CSM_IMPORTER
    -DASSIMP_BUILD_NO_DXF_IMPORTER
    # -DASSIMP_BUILD_NO_GLTF_IMPORTER
    # -DASSIMP_BUILD_NO_GLTF2_IMPORTER
    -DASSIMP_BUILD_NO_HMP_IMPORTER
    -DASSIMP_BUILD_NO_IFC_IMPORTER
    -DASSIMP_BUILD_NO_IRR_IMPORTER
    -DASSIMP_BUILD_NO_IRRMESH_IMPORTER
    -DASSIMP_BUILD_NO_IQM_IMPORTER
    -DASSIMP_BUILD_NO_LWO_IMPORTER
    -DASSIMP_BUILD_NO_LWS_IMPORTER
    -DASSIMP_BUILD_NO_M3_IMPORTER
    -DASSIMP_BUILD_NO_MD2_IMPORTER
    -DASSIMP_BUILD_NO_MD3_IMPORTER
    -DASSIMP_BUILD_NO_MD5_IMPORTER
    -DASSIMP_BUILD_NO_MDC_IMPORTER
    -DASSIMP_BUILD_NO_MDL_IMPORTER
    -DASSIMP_BUILD_NO_MMD_IMPORTER
    -DASSIMP_BUILD_NO_M3D_IMPORTER
    -DASSIMP_BUILD_NO_MS3D_IMPORTER
    -DASSIMP_BUILD_NO_NDO_IMPORTER
    -DASSIMP_BUILD_NO_NFF_IMPORTER
    -DASSIMP_BUILD_NO_OFF_IMPORTER
    -DASSIMP_BUILD_NO_OGRE_IMPORTER
    -DASSIMP_BUILD_NO_OPENGEX_IMPORTER
    -DASSIMP_BUILD_NO_PLY_IMPORTER
    -DASSIMP_BUILD_NO_Q3BSP_IMPORTER
    -DASSIMP_BUILD_NO_Q3D_IMPORTER
    -DASSIMP_BUILD_NO_RAW_IMPORTER
    -DASSIMP_BUILD_NO_SIB_IMPORTER
    -DASSIMP_BUILD_NO_SMD_IMPORTER
    -DASSIMP_BUILD_NO_STEPFILE_IMPORTER
    -DASSIMP_BUILD_NO_STL_IMPORTER
    -DASSIMP_BUILD_NO_TERRAGEN_IMPORTER
    -DASSIMP_BUILD_NO_X3D_IMPORTER
    -DASSIMP_BUILD_NO_XGL_IMPORTER
    -DASSIMP_BUILD_NO_X_IMPORTER
    -DASSIMP_BUILD_NO_XX_IMPORTER
    -DASSIMP_BUILD_NO_STEP_IMPORTER
    -DASSIMP_BUILD_NO_EXPORT
)

if (ANDROID)
add_definitions(
    -DASSIMP_ANDROID_JNIIOSYSTEM
)
endif()

add_definitions(
    -DASSIMP_BUILD_NO_OWN_ZLIB
)

# specify where our headers are
include_directories(${SRC_DIR}/contrib)
include_directories(${SRC_DIR}/contrib/utf8cpp/source)
include_directories(${SRC_DIR}/contrib/irrXML)
include_directories(${SRC_DIR}/contrib/rapidjson/include)
add_definitions( -DRAPIDJSON_HAS_STDSTRING=1)
include_directories(${SRC_DIR})
include_directories(${PRIVATE_HDR_DIR})

# we're building a library
add_library(${TARGET} STATIC ${PRIVATE_HDRS} ${PUBLIC_HDRS} ${SRCS})

if(NOT MSVC)
    target_compile_options(${TARGET}
        PRIVATE -Wno-deprecated-declarations
        PRIVATE -Wno-deprecated-register
        # PRIVATE -Wno-incompatible-pointer-types
        PRIVATE -Wno-ordered-compare-function-pointers
        PRIVATE -Wno-parentheses
        PRIVATE -Wno-sign-compare
        PRIVATE -Wno-strict-aliasing
        PRIVATE -Wno-strict-overflow
        PRIVATE -Wno-tautological-compare
        PRIVATE -Wno-tautological-undefined-compare
        PRIVATE -Wno-undefined-var-template
        PRIVATE -Wno-uninitialized
        PRIVATE -Wno-unused-const-variable
        PRIVATE -Wno-unused-private-field
        PRIVATE -Wno-unused-variable
        PRIVATE -Wno-deprecated-non-prototype # See https://github.com/madler/zlib/issues/633
    )
else()
    target_compile_options(${TARGET} PRIVATE /bigobj)
endif()

if (NOT WIN32)
    target_compile_options(${TARGET} PRIVATE -Wno-address-of-packed-member)
elseif(NOT MSVC)
    # Making this option public because the warnings occurs in public headers of libassimp
    target_compile_options(${TARGET} PUBLIC -Wno-pragma-pack)
    # RapidJSON disables warnings in-code that clang-cl doesnt know about
    target_compile_options(${TARGET} PRIVATE -Wno-unknown-warning-option)
endif()

# specify where the public headers of this library are
target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR})

# and we're linking against the libraries below, importing their public headers
target_link_libraries(${TARGET} LINK_PUBLIC z)

set_target_properties(${TARGET} PROPERTIES FOLDER ThirdParty)
