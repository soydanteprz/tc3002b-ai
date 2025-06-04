import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;

public class A {
    static List<Integer> [] adj;
    static ArrayList<Integer> temp;
    static int mod = (int) 1e9+7;
    static boolean[] vis = new boolean[(int)1e5];
    static List<Integer> l = new ArrayList<>();
    static final Reader s = new Reader();

    public static void main(String[] args) throws IOException {
        int t = s.nextInt();
//        int t=1;
       while(t-->0) {
           int n = s.nextInt();
           int m = s.nextInt();
           int d = s.nextInt();
           System.out.println("YES");
           PriorityQueue<Pair> pq = new PriorityQueue<>();
            for(int i=1;i<=m;i++){
               pq.add(new Pair(i));
            }
            for(int i=0;i<n;i++){
                int v = s.nextInt();
                Pair p = pq.poll();
                p.sum+=v;
                pq.add(p);
                System.out.print(p.id+" ");
            }
           System.out.println();
       }
    }
    static class Pair implements Comparable<Pair> {
        int id;
        long sum=0;
        public Pair(int id) {
            this.id=id;
        }
        
    }
    //Important
    //Dont't stick on same approach
    //Check long
    

    

    

    

    

    

    

}
