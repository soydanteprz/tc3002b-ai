import java.util.*;
import java.io.*;


public class Main{
    public static class Node {
        int val;
        Node left;
        Node right;
        Node(int val)
        {
            this.val=val;
        }
    }
    public static void main(String[] args) throws java.io.IOException {
        Scanner sc=new Scanner(System.in);
        int t=sc.nextInt();
        while(t-->0)
        {
            int n=sc.nextInt();
            int[] arr=new int[n];
            for(int i=0;i<n;++i)
            {
                arr[i]=sc.nextInt();
            }
            Node head=new Node(arr[0]);
            boolean cond=false;
            for(int i=1;i<n;i++)
            {
                int cur=head.val;
                //System.out.println(cur);
                if(arr[i]>cur)
                {
                    if(head.right==null)
                    {
                        Node next=new Node(arr[i]);
                        head.right=next;
                        next.left=head;
                        head=head.right;
                    }
                    else if(head.right.val<arr[i])
                    {
                        cond=true;
                        break;
                    }
                    else if(head.right.val==arr[i])
                    {
                        head=head.right;
                    }
                    else
                    {
                        Node next=new Node(arr[i]);
                        next.right=head.right;
                        head.right.left=next;
                        head.right=next;
                        next.left=head;
                        head=next;
                    }
                }
                else if(arr[i]<cur)
                {
                    if(head.left==null)
                    {
                        Node next=new Node(arr[i]);
                        head.left=next;
                        next.right=head;
                        head=head.left;
                    }
                    else if(head.left.val>arr[i])
                    {
                        cond=true;
                        break;
                    }
                    else if(head.left.val==arr[i])
                    {
                        head=head.left;
                    }
                    else
                    {
                        Node next=new Node(arr[i]);
                        next.left=head.left;
                        head.left.right=next;
                        head.left=next;
                        next.right=head;
                        head=next;
                    }
                }
            }
            if(cond)
                System.out.println("NO");
            else
                System.out.println("YES");
        }
    }
}
