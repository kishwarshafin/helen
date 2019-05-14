//
// Created by Kishwar Shafin on 10/18/18.
//

#ifndef HELEN_PYBIND_API_H
#define HELEN_PYBIND_API_H

#include "local_reassembly/ssw_cpp.h"
#include "local_reassembly/ssw.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
namespace py = pybind11;

PYBIND11_MODULE(HELEN, m) {
        // Alignment CLASS
        py::class_<StripedSmithWaterman::Alignment>(m, "Alignment")
            .def(py::init<>())
            .def_readwrite("best_score", &StripedSmithWaterman::Alignment::sw_score)
            .def_readwrite("best_score2", &StripedSmithWaterman::Alignment::sw_score_next_best)
            .def_readwrite("reference_begin", &StripedSmithWaterman::Alignment::ref_begin)
            .def_readwrite("reference_end", &StripedSmithWaterman::Alignment::ref_end)
            .def_readwrite("query_begin", &StripedSmithWaterman::Alignment::query_begin)
            .def_readwrite("query_end", &StripedSmithWaterman::Alignment::query_end)
            .def_readwrite("ref_end_next_best", &StripedSmithWaterman::Alignment::ref_end_next_best)
            .def_readwrite("mismatches", &StripedSmithWaterman::Alignment::mismatches)
            .def_readwrite("cigar_string", &StripedSmithWaterman::Alignment::cigar_string)
            .def_readwrite("cigar", &StripedSmithWaterman::Alignment::cigar)
            .def("Clear", &StripedSmithWaterman::Alignment::Clear);

        // Filter Class
        py::class_<StripedSmithWaterman::Filter>(m, "Filter")
            .def_readwrite("report_begin_position", &StripedSmithWaterman::Filter::report_begin_position)
            .def_readwrite("report_cigar", &StripedSmithWaterman::Filter::report_cigar)
            .def_readwrite("score_filter", &StripedSmithWaterman::Filter::score_filter)
            .def_readwrite("distance_filter", &StripedSmithWaterman::Filter::distance_filter)
            .def(py::init<>())
            .def(py::init<const bool&, const bool&, const uint16_t&, const uint16_t&>());

        // Aligner Class
        py::class_<StripedSmithWaterman::Aligner>(m, "Aligner")
            .def(py::init<>())
            .def(py::init<const uint8_t&, const uint8_t&, const uint8_t&, const uint8_t&>())
            .def("SetReferenceSequence", &StripedSmithWaterman::Aligner::SetReferenceSequence)
            .def("Align_cpp", &StripedSmithWaterman::Aligner::Align_cpp);
}
#endif //HELEN_PYBIND_API_H
