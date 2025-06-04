import java.util.*;
//import java.util.Stack;
import java.io.BufferedReader;
import java.io.IOException;
//import java.io.*;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.SortedSet;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;
import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.BufferedWriter;
import java.io.Writer;
import java.io.OutputStreamWriter;
import java.util.InputMismatchException;
import java.io.IOException;
import java.io.InputStream;
public class Main {


    

    

     static class obj{
           long a1;
           long a2;
           //int a3;
           obj(long a1,long a2){
               this.a1=a1;
               this.a2=a2;
              // this.a3=a3;
           }
       }
       static  class sortby implements Comparator<obj>{
           
       }
    
    
    
    

    //static ArrayList<Integer>[] a;
   // static int[] vis;
    //static int[] col;
  /*  static boolean dfs(int i,int c){
        vis[i]=1;
        col[i]=c;
        //System.out.println(i+" "+c);
        for(int j:a[i]){
            if(vis[j]==0) {
                if (dfs(j, c ^ 1) == false)
                    return false;
            }
            else
                if(col[i]==col[j])
                    return false;
        }
        return true;
    }*/
   static int[] m;
    
   static long[][] dis;
   // static long d=1000000007;
   
    public static void main(String[] args) {
        FastReader s = new FastReader();
        //InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        PrintWriter out = new PrintWriter(outputStream);
        //Scanner s=new Scanner(System.in);
         int t = s.nextInt();

        while(t-->0) {
            int n = s.nextInt();
            int m = s.nextInt();
            int x=s.nextInt();
            long[] a = new long[n];
            for(int i=0;i<n;i++){
                a[i]=s.nextInt();
            }
           out.println("YES");
            PriorityQueue<obj> p = new PriorityQueue<>(new sortby());
            for (int i = 0; i <m; i++) {
             p.add(new obj(a[i],i+1));
             out.print(i+1+" ");
            }
            for(int i=m;i<n;i++){
                obj o=p.poll();
                p.add(new obj(o.a1+a[i],o.a2));
                out.print(o.a2+" ");
            }
            out.println();
        }
       out.close();

}}
