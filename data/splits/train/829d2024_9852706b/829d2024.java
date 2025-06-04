import java.util.*;
import java.io.*;


public class Main{
    public static class Element implements Comparable<Element>{
        public int key;
        public int value;
        Element(int k, int v)
        {
            key=k;
            value=v;
        }

        
    }
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        int t=sc.nextInt();
        while(t-->0)
        {
            int n=sc.nextInt();
            int m=sc.nextInt();
            int x=sc.nextInt();
            int[] arr=new int[n];
            for(int i=0;i<n;++i)
                arr[i]=sc.nextInt();
            PriorityQueue<Element> pq=new PriorityQueue<>();
            for(int i=1;i<=m;++i)
            {
                pq.add(new Element(i,0));
            }
            System.out.println("YES");
            for(int j=0;j<n;j++)
            {
                Element cur = pq.poll();
                System.out.print(cur.key+" ");
                cur.value+= arr[j];
                pq.add(cur);
            }
            System.out.println();
        }
    }
}
