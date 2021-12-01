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
  :parameters (?parcel ?vehicle ?spot)
  :precondition (and 
                    (or 
                        (and (DIST-CENTER ?spot)
                             ;(not(AREA ?spot))
                             (long-range-vehicle ?vehicle)
                             ;(not(short-range-vehicle) ?vehicle)
                        )
                        (and (DIST-CENTER ?spot)
                             ;(not(AREA ?spot))
                             ;(not(long-range-vehicle ?vehicle))
                             (short-range-vehicle ?vehicle)
                        )
                        ;(and (not(DIST-CENTER ?spot)) ;case of long-vehicle in AREA, impossible
                        ;     (AREA ?spot)
                        ;     (long-range-vehicle ?vehicle)
                        ;     (not(short-range-vehicle ?vehicle))
                        ;)
                        (and ;(not(DIST-CENTER ?spot))
                             (AREA ?spot)
                             ;(not(long-range-vehicle ?vehicle))
                             (short-range-vehicle ?vehicle)
                        )
                    )
                    (PARCEL ?parcel)
                    (VEHICLE ?vehicle)
                    (is-parcel-at ?parcel ?spot)
                    (is-vehicle-at ?vehicle ?spot)
                )
  :effect (and 
              (is-parcel-in-vehicle ?parcel ?vehicle)
              (not(is-parcel-at ?parcel ?spot))
          ); WRITE HERE THE CODE FOR THIS ACTION
  )
  
  ; Parcel x is unloaded from vehicle y in area/dc z if the parcel x is in the 
  ; vehicle y and the vehicle y is at z.
  ; As a result, parcel x is not in vehicle y anymore and the parcel x is at z
  ; Parameters
  ; - x: parcel
  ; - y: vehicle
  ; - z: area or distribution center
  (:action unload-parcel
  :parameters (?parcel ?vehicle ?spot)
  :precondition (and (or 
                        (and (DIST-CENTER ?spot)
                             (not(AREA ?spot))
                             (long-range-vehicle ?vehicle)
                             (not(short-range-vehicle ?vehicle))
                        )
                        (and (DIST-CENTER ?spot)
                             (not(AREA ?spot))
                             (not(long-range-vehicle ?vehicle))
                             (short-range-vehicle ?vehicle)
                        )
                        (and (not(DIST-CENTER ?spot))
                             (AREA ?spot)
                             (not(long-range-vehicle ?vehicle))
                             (short-range-vehicle ?vehicle)
                        )
                    )
                    (PARCEL ?parcel)
                    (VEHICLE ?vehicle)
                ;    (or (DIST-CENTER ?spot)
  ;                      (AREA ?spot))
                    (is-vehicle-at ?vehicle ?spot)
                    (is-parcel-in-vehicle ?parcel ?vehicle)
                    (not(is-parcel-at ?parcel ?spot))
            )
  :effect (and 
                (not(is-parcel-in-vehicle ?parcel ?vehicle))
                (is-parcel-at ?parcel ?spot)
            )
  )
  ; Long-distance travel, i.e. between distribution centers x and y by a 
  ; long-range vehicle z if x and y are connected.
  ; As a result, vehicle z is at y.
  ; Parameters
  ; - x: dc from
  ; - y: dc to
  ; - z: long-range-vehicle
  (:action travel-long
  :parameters (?spot1 ?spot2 ?vehicle)
  :precondition (and 
                    (DIST-CENTER ?spot1)
                    (DIST-CENTER ?spot2)
                    (VEHICLE ?vehicle)
                    (long-range-vehicle ?vehicle)
                    (connected ?spot1 ?spot2)
                    (is-vehicle-at ?vehicle ?spot1)
                )
  :effect (and (is-vehicle-at ?vehicle ?spot2)
               (not(is-vehicle-at ?vehicle ?spot1))
               (long-range-vehicle ?vehicle)
          )
  ); WRITE HERE THE CODE FOR THIS ACTION

  ; Short-distance travel, i.e. not between distribution centers, by a 
  ; short-range vehicle z if x and y are connected.
  ; As a result, vehicle z is at y.
  ; Parameters
  ; - x: area/dc from
  ; - y: dc/area to
  ; - z: short-range-vehicle
  (:action travel-short
  :parameters (?spot1 ?spot2 ?vehicle)
  :precondition (and 
                     (or (and
                              (AREA ?spot1)
                              (AREA ?spot2)
                         )
                         (and 
                              (DIST-CENTER? ?spot1)
                              (AREA ?spot2)
                         )
                         (and 
                              (AREA ?spot1)
                              (DIST-CENTER ?spot2)
                         )
                     )
                     (VEHICLE ?vehicle)
                     (short-range-vehicle ?vehicle)
                     (connected ?spot1 ?spot2)
                     (is-vehicle-at ?vehicle ?spot1)
                )
  :effect (and (is-vehicle-at ?vehicle ?spot2)
               (not(is-vehicle-at ?vehicle ?spot1))
               (short-range-vehicle ?vehicle)
          )
    ; WRITE HERE THE CODE FOR THIS ACTION
  )
)