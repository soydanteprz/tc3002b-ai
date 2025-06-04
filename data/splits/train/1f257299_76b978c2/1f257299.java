import java.lang.reflect.Array;
import java.util.*;
import java.io.*;

//import static com.sun.tools.javac.jvm.ByteCodes.swap;




public class fastTemp {


    static FastScanner fs = null;

    public static void main(String[] args) {
        fs = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);

        int t = fs.nextInt();
        while (t-- > 0) {
            int n = fs.nextInt();
           int p[] = new int[n];
           Arrays.fill(p,-1);
           for(int i=0;i<n;i++){
               if(p[i]!=-1){
                   continue;
               }
               System.out.println("? "+(i+1));
               int ans = (i+1);
               int x = fs.nextInt();
               while(p[x-1]==-1){
                   System.out.println("? "+(i+1));
                   int last = fs.nextInt();
                   p[x-1] = last;
                   x = last;
               }

           }
            System.out.print("! ");
           for(int i=0;i<n;i++){
               System.out.print(p[i]+" ");
           }

            System.out.flush();

        }
    }

    
    

    

    

    

    

    

    

    

    // Returns n^(-1) mod p
    

    // Returns nCr % p using Fermat's
    // little theorem.

    


//public static int dijkstra(int src , int dist[] ){
//
//PriorityQueue<Pair> q = new PriorityQueue<>();
//q.add(new Pair(1,0));
//
//while(q.size()>0){
//
//    Pair rem = q.remove();
//    for(Pair x:graph[rem.y]){
//        if(dist[x.y]>dist[rem.y]+x.wt){
//            dist[x.y] = dist[rem.y] + x.wt;
//            q.add(new Pair(x.y,dist[x.y]));
//        }
//    }
//
//}
//
//return dist[dist.length-1];
//
//}
    // T --> O(n) && S--> O(d)




    

    



    


    
    
    // ------------------------------------------swap----------------------------------------------------------------------

    


    //-------------------------------------------seiveOfEratosthenes----------------------------------------------------


    



    //------------------------------------------- power------------------------------------------------------------------



    
    

    //---------------------------------------EXTENDED EUCLID ALGO--------------------------------------------------------
//    public static class Pair{
//        int x;
//        int y;
//        public Pair(int x,int y){
//            this.x = x;
//            this.y = y ;
//        }
//    }

//    public static Pair Euclid(int a,int b){
//
//        if(b==0){
//            return new Pair(1,0);  // answer of x and y
//        }
//
//        Pair dash = Euclid(b,a%b);
//
//        return new Pair(dash.y , dash.x - (a/b)*dash.y);
//
//
//    }
//--------------------------------GCD------------------GCD-----------GCD--------------------------------------------

    
    

// This is an extension of method 2. Instead of moving one by one, divide the array in different sets
//where number of sets is equal to GCD of n and d and move the elements within sets.
//If GCD is 1 as is for the above example array (n = 7 and d =2), then elements will be moved within one set only, we just start with temp = arr[0] and keep moving arr[I+d] to arr[I] and finally store temp at the right place.
//Here is an example for n =12 and d = 3. GCD is 3 and


//     void leftRotate(int arr[], int d, int n)
//     {
//         /* To handle if d >= n */
//         d = d % n;
//         int i, j, k, temp;
//         int g_c_d = gcd(d, n);
//         for (i = 0; i < g_c_d; i++) {
//             /* move i-th values of blocks */
//             temp = arr[i];
//             j = i;
//             while (true) {
//                 k = j + d;
//                 if (k >= n)
//                     k = k - n;
//                 if (k == i)
//                     break;
//                 arr[j] = arr[k];
//                 j = k;
//             }
//             arr[j] = temp;
//         }
//     }


}
