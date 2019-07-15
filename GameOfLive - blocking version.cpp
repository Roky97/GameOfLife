///////////////////////
//  ROCCO PALERMITI  //
// blocking version  //
///////////////////////

#include <iostream>
using namespace std;
#include <mpi.h>
#include <stdlib.h>
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>

//GESTIONE GRAFICA
const int FPS=60; //frame al secondo
const int cicli=10000; //numero di cicli da eseguire

//RISOLUZIONE
// int w=1920; //fullHD
// int h=1080;

int w=2560; //MacBook Pro 13'
int h=1600;

int dimQuad=10; // dimenzione del lato una singola casella in pixel. (5 o 10 o 20)

//se la risoluzione del monitor è quella del MacBook Pro 13' impostare questi come paramentro:
//h1 sara' la dimensione VERTICALE della matrice e w1 di quella ORIZZONTALE

// const int h1=320; //se dimQuad è 5
// const int w1=512; //se dimQuad è 5

const int h1=160; //se dimQuad è 10
const int w1=256; //se dimQuad è 10

// const int h1=80; //se dimQuad è 20
// const int w1=128; //se dimQuad è 20

//se la risoluzione del monitor è in fullHD impostare qesti come paramentro:
// const int h1=216; //se dimQuad è 5
// const int w1=384; //se dimQuad è 5

// const int h1=108; //se dimQuad è 10
// const int w1=192; //se dimQuad è 10

// const int h1=54; //se dimQuad è 20
// const int w1=96; //se dimQuad è 20

void restart(bool matrice[h1][w1],const int& w, const int& h) //funzione che reinizializza la matrice in maniera casuale
{
    int num;
    for(int i=0; i<h; i++)
    {
        for(int j=0; j<w; j++)
        {
            num=rand()%2;
            if(num==0)
                matrice[i][j]=0;
            else
                matrice[i][j]=1;
        }
    }
}

void adiacenti(int x, int y, bool m[h1][w1], bool ad[h1][w1], int contAd, const int& w, const int& h, int id) // calcola le adiacenze per ogni punto della matrice
{
    int i_min = (x - 1) < 0 ? x : x - 1;
    int j_min = (y - 1) < 0 ? y : y - 1;

    int i_max = x + 1 < h ? x + 1 : x;
    int j_max = y + 1 < w ? y + 1 : y;

    for(int a = i_min; a <= i_max; a++)
    {
        for(int b = j_min; b <= j_max; b++)
        {
            if(m[a][b])
                contAd++;
        }
    }

    if(m[x][y])
        contAd--;

    ad[x][y] = (contAd == 3 || (contAd == 2 && m[x][y]));

    contAd=0;
}

void evolveMatrice(bool a[h1][w1], const int & w, const int & h, const int & nThreads, int id) //funzione principale
{
    MPI_Status stat;
    MPI_Datatype totRighe;
    MPI_Datatype riga;
    int nRows=(h/nThreads);

    MPI_Type_contiguous(nRows*w, MPI_CXX_BOOL, &totRighe);
    MPI_Type_contiguous(w, MPI_CXX_BOOL, &riga);

    MPI_Type_commit(&totRighe);
    MPI_Type_commit(&riga);

    bool ad[h1][w1]; //altra matrice di appoggio
    int contAd=0; //contatore di adiacenze rispetto ad un punto sulla matrice

    for( int i=1; i<nThreads; i++) //invia le porzioni di patrice per ogni thread
    {

        int pos = nRows*i;
        MPI_Send(&(a[pos][0]), 1, totRighe, i, 0, MPI_COMM_WORLD);
    }

    for(int i=1; i<nThreads; i++) //per ogni thread manda i bordi delle matrice "ghost points" necessari per determinare le adiacence esatte per ogni punto della matrice
    {
        if(i<nThreads-1)
        {
            int posIn=(nRows*i) -1 ;
            MPI_Send(&(a[posIn][0]), 1, riga, i, 0, MPI_COMM_WORLD); // ghost point inizio

            int posFin=nRows*(i+1);
            MPI_Send(&(a[posFin][0]), 1, riga, i, 0, MPI_COMM_WORLD); // ghost point fine
        }
        else if(i==nThreads-1)
        {
            int posIn=(nRows*i) -1 ;

            MPI_Send(&(a[posIn][0]), 1, riga, i, 0, MPI_COMM_WORLD); // ghost point inizio
        }
    }

    for(int i=0; i<nRows; i++) //il master evolve la sua porzione di matrice
    {
        for(int j=0; j<w; j++)
        {
            adiacenti(i,j,a,ad,contAd,w,h,id);
        }
    }

    for( int i=1; i<nThreads; i++) //per ogni thread ricevo la porzione di matrice evoluta
    {
        int nElements=nRows*w;
        int pos = nRows*i;

        MPI_Recv(&(a[pos][0]), nElements, MPI_CXX_BOOL, i, 0, MPI_COMM_WORLD, &stat);

    }

    for(int i=0; i<nRows; i++) //copia la matrice di appoggio evoluta dal master sulla matrice originale
    {
        for(int j=0; j<w; j++)
        {
            a[i][j]=ad[i][j];
        }
    }

    MPI_Type_free(&totRighe);
    MPI_Type_free(&riga);
}

void init()
{
    al_init();
    al_install_keyboard();
    al_install_mouse();
}

int main(int argc, char **argv) {

    w/=dimQuad; //dimensione orizzontale della matrice che dovrebbe equivalere ad 'w1'
    h/=dimQuad; //dimensione verticale della matrice che dovrebbe equivalere ad 'h1'

    MPI_Datatype totRighe; //datatype per mandare e ricevere le righe della matrice
    MPI_Status stat;

    bool doexit = false; //booleana di uscita dal gioco

    MPI_Init(&argc, &argv);

    int nThreads; //numero dei thread
    MPI_Comm_size(MPI_COMM_WORLD, &nThreads);

    int id; //ID del thread
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int nRows=(h/nThreads); // numero di righe da mandare/spedire in base alla dimensione verticale della matrice

    MPI_Type_contiguous(nRows*w, MPI_CXX_BOOL, &totRighe); //creazione del datatype
    MPI_Type_commit(&totRighe);

    if(id==0)  //CODICE ESEGUITO SOLO DAL MASTER
    {
        init(); // inizializzo le funzionalita' di allegro

        ALLEGRO_DISPLAY *display = NULL; //dichiarazione del display
        ALLEGRO_EVENT_QUEUE *event_queue = NULL; //dichiarazione della pila di eventi
        ALLEGRO_TIMER *timer = NULL; //dichiarazione del timer di aggiornamento del display
        bool redraw = true;

        bool m[h1][w1]; //matrice principale

        srand(time(0));

        timer = al_create_timer(1.0 / FPS); //creazione del timer

        al_set_new_display_flags(ALLEGRO_FULLSCREEN_WINDOW); //settaggio modalita schermo pieno
        display = al_create_display(w, h); //crezione del display

        event_queue = al_create_event_queue(); //creazione della pila

        al_register_event_source(event_queue, al_get_display_event_source(display));
        al_register_event_source(event_queue, al_get_keyboard_event_source());
        al_register_event_source(event_queue, al_get_timer_event_source(timer));
        al_register_event_source(event_queue, al_get_mouse_event_source());


        int num; //variabile utile per settare la matrice
        for(int i=0; i<h; i++) //INIZIALIZZO LA MATRICE
        {
            for(int j=0; j<w; j++)
            {
                num=rand()%2;
                if((i+j)%2==0)
                    m[i][j]=0;
                else
                    m[i][j]=1;
            }
        }

        al_start_timer(timer);
        bool finish=false; //varibile utile per sincronizzare la chiusura del gioco tra il pocesso master e gli slaves

        double tempo; //contatone del tempo impiegato per eseguire la parte logica del gioco
        int contCicli=0; //contatore dei cicli del gioco

        int r=255, g=0, b=0; //colore RGB delle celle sul display

        while(!doexit && contCicli<cicli)
        {
            ALLEGRO_EVENT ev;
            al_wait_for_event(event_queue, &ev);
            if(ev.type == ALLEGRO_EVENT_DISPLAY_CLOSE)
            {
                finish=true;

            }

            if(ev.type == ALLEGRO_EVENT_TIMER)
            {
                double start =MPI_Wtime();

                evolveMatrice(m,w,h, nThreads, id); //evolve la parte di matrice dedicata al master e spedisce le altri porzioni agli slaves
                double end =MPI_Wtime();

                tempo+=end-start;

                if(finish)
                    doexit=true;
                contCicli++;

                if(contCicli==cicli-1)
                    finish=true;

                for(int i=1; i<nThreads; i++)
                    MPI_Send(&doexit, 1, MPI_CXX_BOOL, i, 1, MPI_COMM_WORLD); //spedisco la booleana del ciclo di gioco agli slaves

                redraw=true;
            }

            if( ALLEGRO_EVENT_MOUSE_BUTTON_DOWN) //quando si passa con il mouse sopra una cella quest'ultima diventera' viva
            {
                if((ev.mouse.x/dimQuad)>=0 && (ev.mouse.x/dimQuad)<w && (ev.mouse.y/dimQuad)>=0 && (ev.mouse.y/dimQuad)<h)
                    m[(ev.mouse.y/dimQuad)][(ev.mouse.x/dimQuad)]=1;
            }

            if(ev.type == ALLEGRO_EVENT_KEY_DOWN)
            {
                switch(ev.keyboard.keycode)
                {
                case ALLEGRO_KEY_R: //se premuto il tasto 'R' la matrice si reinizializzera' in maniera casuale

                    restart(m, w, h);
                    break;
                case ALLEGRO_KEY_ESCAPE: //se premuto il tasto 'esc' il gioco si chiudera'

                    finish=true;
                    redraw=false;
                    break;
                }
            }

            if(redraw && al_is_event_queue_empty(event_queue)) //mostra su display la matrice
            {
                // al_clear_to_color(al_map_rgb(r++,g--,b++));
                al_clear_to_color(al_map_rgb(0,0,0));

                for(int i=0; i<h; i++)
                    for(int j=0; j<w; j++)
                    {
                        if(m[i][j])
                        {
                            al_draw_rectangle(j*dimQuad, i*dimQuad, (j*dimQuad)+dimQuad, (i*dimQuad)+dimQuad, al_map_rgb(0,0,0), 0);
                            al_draw_filled_rectangle(j*dimQuad, i*dimQuad, (j*dimQuad)+dimQuad, (i*dimQuad)+dimQuad, al_map_rgb(r,g,b));
                        }
                    }

                al_flip_display();
                al_rest(0.017);
                redraw=false;
            }
        }
        al_destroy_timer(timer);
        al_destroy_display(display);
        al_destroy_event_queue(event_queue);

        cout<<"Tempo in "<<contCicli<<" cicli: "<<tempo<<endl;
    }

    if(id!=0) //CODICE ESEGUITO DAI PROCESSI SLAVES
    {
        while(!doexit)
        {
            int nRows=((h/nThreads)); //numero di righe da mandare/spedire
            int nElements=nRows*w; //numero totale di elementi mandare/spedire

            bool subMatrix[nRows][w]; //matrice locale
            bool ghostIn[w]; //array di goost points che rappresenta la riga finale della matrice locale precedente
            bool ghostFin[w]; //array di goost points che rappresenta la riga iniziale della matrice locale successiva

            MPI_Recv(&subMatrix[0][0], nElements, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, &stat); //riceve la porzione di matrice dal master

            if(id<nThreads-1)
            {
                MPI_Recv(ghostIn, w, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, &stat); // ghost point inizio

                MPI_Recv(ghostFin, w, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, &stat); // ghost point fine
            }
            else if(id==nThreads-1)
            {
                MPI_Recv(ghostIn, w, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, &stat); // ghost point inizio
            }

            bool ad[nRows][w]; //altra matrice locale di appoggio su cui si applicano le modifiche
            int contAd=0;

            for(int i=0; i<nRows; i++)
                for(int j=0; j<w; j++)
                {
                    int i_min = (i - 1) < 0 ? i : i - 1;
                    int j_min = (j - 1) < 0 ? j : j - 1;

                    int i_max = i + 1 < nRows ? i + 1 : i;
                    int j_max = j + 1 < w ? j + 1 : j;

                    for(int a = i_min; a <= i_max; a++)
                    {
                        for(int b = j_min; b <= j_max; b++)
                        {
                            if(subMatrix[a][b])
                                contAd++;
                        }
                    }

                    if(subMatrix[i][j])
                        contAd--;

                    if(i==0)
                    {
                        for(int f=j_min; f<=j_max; f++)
                            if(ghostIn[f])
                                contAd++;
                    }
                    if(i==(nRows-1) && id<(nThreads-1))
                    {
                        for(int f=j_min; f<=j_max; f++)
                            if(ghostFin[f])
                                contAd++;
                    }

                    ad[i][j] = (contAd == 3 || (contAd == 2 && subMatrix[i][j]));

                    contAd=0;
                }

            MPI_Send(&ad[0][0], 1, totRighe, 0, 0, MPI_COMM_WORLD); //spedisco la porzione di matrice aggiornata al master
            MPI_Recv(&doexit, 1, MPI_CXX_BOOL, 0, 1, MPI_COMM_WORLD, &stat); // ricevo la booleana del ciclo di gioco dal master
        }
    }

    MPI_Type_free(&totRighe);
    MPI_Finalize();

    return 0;
}
