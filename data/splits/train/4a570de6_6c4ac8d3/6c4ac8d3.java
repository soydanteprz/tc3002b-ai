import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Scanner;
import java.io.*;


public class ieee1{

    
    public static void main(String[] args) {
           Scanner scn=new Scanner(System.in);
           int t=scn.nextInt();
           while(t-->0){
               HashMap<Integer,Integer> map=new HashMap<>();
              
               int b=scn.nextInt();
               int m=scn.nextInt();
               int x=scn.nextInt();
               int[] arr=new int[b];
               PriorityQueue<Node> pq=new PriorityQueue<>(new pqc());
               for(int i=0;i<b;i++){
                   int ele=scn.nextInt();
                   
                  
                   arr[i]=ele;
               }
               for(int i=1;i<=m;i++){
                   pq.add(new Node(i,0));
               }
               System.out.println("YES");
               for(int i=0;i<arr.length;i++){
                   int ele=arr[i];
                   Node n=pq.poll();
                   System.out.print(n.ind+" ");
                   n.data+=ele;
                   pq.add(n);
               }
               System.out.println();
              
              
    }
}
    public static class Node{
        int ind;
        int data;
        Node(int ind,int data){
            this.ind=ind;
            this.data=data;
        }
    }
    public static class pqc implements Comparator<Node>{
        
        
    }

}
