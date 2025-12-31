from sqlalchemy import Column, Integer, String, Date, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class Film(Base):
    __tablename__ = "films"

    film_id = Column(Integer, primary_key=True, index=True)
    titre = Column(String, nullable=False)
    date_sortie = Column(Date, nullable=True)

    reviews = relationship("Review", back_populates="film")

class Review(Base):
    __tablename__ = "reviews"

    review_id = Column(String, primary_key=True, index=True)
    film_id = Column(Integer, ForeignKey("films.film_id"))
    auteur = Column(String)
    contenu = Column(String)
    edited = Column(Integer, default=0)

    film = relationship("Film", back_populates="reviews")


class Users(Base):
    __tablename__ = "users"

    users_id = Column(String, primary_key=True, index=True)
    mail = Column(String, nullable=False)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)