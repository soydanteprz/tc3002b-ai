import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Main {
    public static void main(String[] args) {
        FastScanner sc = new FastScanner();
        int t=sc.nextInt();
        while (t-->=1) {
            int n = sc.nextInt(),m=sc.nextInt(), x = sc.nextInt();
            int a[] = sc.readArray(n);
            PriorityQueue<Pair> pq=new PriorityQueue<>();
            for (int i=0;i<m;i++){
                pq.add(new Pair(0,i+1));
            }
            int ans[]= new int[n];

            System.out.println("YES");
            for (int i=0;i<n;i++){
                Pair e=pq.peek();
                pq.remove(e);
                e.a+=a[i];
                pq.add(e);
                ans[i]=e.b;
//                System.out.print((e.b)%m+1+" ");
            }
            printArray(ans);
            System.out.println();


        }
    }



//        out.flush();



    //------------------------------------if-------------------------------------------------------------------------------------------------------------------------------------------------
    //sieve
    
    
    
    static class Pair implements Comparable<Pair> {
        int a;
        int b;

        public Pair(int a, int b) {
            this.a = a;
            this.b = b;
        }
        //        to sort first part
//        public int compareTo(Pair other) {
//            if (this.a == other.a) return other.b > this.b ? -1 : 1;
//            else if (this.a > other.a) return 1;
//            else return -1;
//        }
//        public int compareTo(Pair other) {
//            if (this.b == other.b) return 0;
//            if (this.b < other.b) return 1;
//            else return -1;
//        }
        //sort on the basis of first part only
        

    }
    
    
    static int mod =(int)(1e9+7);
    static long mod(long x) {
        return ((x % mod + mod) % mod);
    }
    static long add(long x, long y) {
        return mod(mod(x) + mod(y));
    }
    
    //Fast Power(logN)
    
    
    
    static void printArray(int a[]) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + " ");
        }
    }
    
    
    
    
    
    
}
