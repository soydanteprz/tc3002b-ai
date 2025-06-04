import java.io.*;
import java.util.*;

public class A {
    //--------------------------INPUT READER---------------------------------//

    //-----------------------------------------------------------------------//

    //---------------------------PRINTER-------------------------------------//

    //--------------------------VARIABLES------------------------------------//
    static FastReader sc = new FastReader();
    static OutputStream outputStream = System.out;
    //-----------------------------------------------------------------------//

    //--------------------------ADMIN_MODE-----------------------------------//
    private static void ADMIN_MODE() throws IOException {
        if (System.getProperty("ONLINE_JUDGE") == null) {
            sc = new FastReader(new FileInputStream("input.txt"));
        }
    }
    //-----------------------------------------------------------------------//

    //----------------------------START--------------------------------------//
    public static void main(String[] args)
            throws IOException {

        ADMIN_MODE();

        int t = sc.ni();while(t-->0)
            solve();


    }

    static int[] res;
    static List<List<Integer>> graph;
    static boolean two = true;
    static HashMap<pr<Integer, Integer>, Integer> hm;

    static void solve() throws IOException {
        int n = sc.ni();
        graph = new ArrayList<>();
        res = new int[n-1];
        hm = new HashMap<>();
        for(int i = 0; i <= n; i++) {
            graph.add(new ArrayList<>());
        }
        boolean f = false;
        for(int i = 0; i < n-1; i++) {
            int a = sc.ni(), b = sc.ni();
            graph.get(a).add(b);
            graph.get(b).add(a);
            if(graph.get(a).size() > 2 || graph.get(b).size() > 2) f = true;
            hm.put(new pr<>(a, b), i);
            hm.put(new pr<>(b, a), i);
        }

        if(f) {
            System.out.println(-1);
            return;
        }

        int one = 0;
        for(int i = 0; i < n; i++) {
            if(graph.get(i).size() == 1) {
                one = i;
                break;
            }
        }
        dfs(one, -1);
        for(int i: res) {
            System.out.println(i+" ");
        }
        System.out.println();
    }

    static void dfs(int at, int pt) {
        List<Integer> li = graph.get(at);
        if(pt != -1) {
            res[hm.get(new pr<>(at, pt))] = two?2:3;
            two = !two;
        }
        for(int to: li) {
            if(to == pt) continue;
            dfs(to, at);
        }
    }

    static class pr <T, V> {
            T a; V b;

            public pr(T a, V b) {
                this.a = a;
                this.b = b;
            }

            

            
        }

}
