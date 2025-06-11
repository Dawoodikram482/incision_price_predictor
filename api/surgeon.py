from flask import Blueprint, jsonify, request
from extensions import db
from models import Surgeon, Procedure, ProcedureSurgeon

#all surgeons of a procedure
surgeon_bp = Blueprint("procedure", __name__)

@surgeon_bp.route("/surgeons", methods=["GET"])
def get_surgeons_by_procedure():
    # this is the ID coming from CSV
    csv_proc_id = request.args.get("procedure_id")

    if not csv_proc_id:
        return jsonify({"error": "Missing 'procedure_id' query parameter"}), 400

    try:
        csv_proc_id = int(csv_proc_id)
    except ValueError:
        return jsonify({"error": "'procedure_id' must be an integer"}), 400

    # join Surgeon → ProcedureSurgeon → Procedure, but filter on Procedure.procedure_id
    surgeons = (
        Surgeon.query
        .join(ProcedureSurgeon, Surgeon.id == ProcedureSurgeon.surgeon_id)
        .join(Procedure, Procedure.id == ProcedureSurgeon.procedure_id)
        .filter(Procedure.procedure_id == csv_proc_id)
        .all()
    )

    if not surgeons:
        return jsonify({"message": "No surgeons found for that procedure ID."}), 404

    result = [
        {
            "id": s.id,
            "name": s.name,
            "speciality_id": s.speciality_id,
            "created_at": s.created_at.isoformat()
        }
        for s in surgeons
    ]

    return jsonify(result), 200



