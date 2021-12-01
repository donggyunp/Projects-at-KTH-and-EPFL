;; Domain definition
(define (domain delivery-domain)
  
  ;; Predicates: Properties of objects that we are interested in (boolean)
  (:predicates
    (DIST-CENTER ?x) ; True if x is a distribution center
    (AREA ?x) ; True if x is an area
    (PARCEL ?x) ; True if x is a parcel
    (VEHICLE ?x) ; True if x is a method of transportation
    (long-range-vehicle ?x) ; True if x is a long-range vehicle
    (short-range-vehicle ?x) ; True if x is a short-range vehicle
    (connected ?x ?y) ; True if dc/area x is connected to dc/area y
    (is-parcel-at ?x ?y) ; True if parcel x is at dc/area y
    (is-vehicle-at ?x ?y) ; True if vehicle x is at area/dc y
    (is-parcel-in-vehicle ?x ?y) ; True if parcel x is in vehicle y
  )
  ;; Actions: Ways of changing the state of the world
  
  ; Parcel x is loaded into vehicle y if both are in the same area/dc z.
  ; As a result, parcel x is in vehicle y and not at z anymore.
  ; Parameters
  ; - x: parcel
  ; - y: vehicle
  ; - z: area or distribution center
  (:action load-parcel
    ; WRITE HERE THE CODE FOR THIS ACTION
  )
  
  ; Parcel x is unloaded from vehicle y in area/dc z if the parcel x is in the 
  ; vehicle y and the vehicle y is at z.
  ; As a result, parcel x is not in vehicle y anymore and the parcel x is at z
  ; Parameters
  ; - x: parcel
  ; - y: vehicle
  ; - z: area or distribution center
  (:action unload-parcel
    ; WRITE HERE THE CODE FOR THIS ACTION
  )

  ; Long-distance travel, i.e. between distribution centers x and y by a 
  ; long-range vehicle z if x and y are connected.
  ; As a result, vehicle z is at y.
  ; Parameters
  ; - x: dc from
  ; - y: dc to
  ; - z: long-range-vehicle
  (:action travel-long
    ; WRITE HERE THE CODE FOR THIS ACTION
  )

  ; Short-distance travel, i.e. not between distribution centers, by a 
  ; short-range vehicle z if x and y are connected.
  ; As a result, vehicle z is at y.
  ; Parameters
  ; - x: area/dc from
  ; - y: dc/area to
  ; - z: short-range-vehicle
  (:action travel-short
    ; WRITE HERE THE CODE FOR THIS ACTION
  )
  
  
)